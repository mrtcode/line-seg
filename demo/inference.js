// -----------------------------------------------------------------------------
// Layout model: ONNX Runtime + CRF-based block detection
// Exposes a single async function:
//
//   window.inference(vectors, lines) -> Promise<Block[]>
//
// Where `vectors` is an array of feature rows (number[NUM_FEATURES]),
// and `lines` is the corresponding array of text lines.
// -----------------------------------------------------------------------------

// ----------------- Resolve paths relative to this script -----------------

/**
 * Determine the base URL for loading the model files (model.onnx, model.crf.json)
 * relative to the current script tag.
 */
const CURRENT_SCRIPT = document.currentScript;

/**
 * Base URL (ending with a slash) from which model assets are loaded.
 * This is resolved relative to the script's own URL when possible,
 * otherwise it falls back to the page URL.
 * @type {string}
 */
const SCRIPT_BASE_URL = (() => {
	if (!CURRENT_SCRIPT || !CURRENT_SCRIPT.src) {
		// Fallback: use the page URL directory
		const url = new URL(window.location.href);
		url.pathname = url.pathname.replace(/[^/]*$/, '');
		return url.toString();
	}

	const url = new URL(CURRENT_SCRIPT.src, window.location.href);
	url.pathname = url.pathname.replace(/[^/]*$/, '');
	return url.toString();
})();

// --------------------------- Global state ---------------------------------

// `ort` is provided globally by onnxruntime-web (loaded via a separate <script> tag).
let ortGlobal = window.ort;

/**
 * Cached model bundle:
 * {
 *   session: ort.InferenceSession,
 *   io: { featureInputName, maskInputName, outputName },
 *   crf: {
 *     numTags: number,
 *     transitions: Float32Array,
 *     start: Float32Array,
 *     end: Float32Array
 *   }
 * }
 * @type {null | {
 *   session: any,
 *   io: { featureInputName: string, maskInputName: string, outputName: string },
 *   crf: { numTags: number, transitions: Float32Array, start: Float32Array, end: Float32Array }
 * }}
 */
let model = null;

/**
 * In-flight initialization promise for the model bundle, to avoid duplicate
 * initialization when inference is called multiple times concurrently.
 * @type {Promise<any> | null}
 */
let modelPromise = null;

// ---------------- ONNX Runtime loading (browser, global ort) --------------

/**
 * Ensure that the global `ort` instance is available and configured.
 * Throws if onnxruntime-web has not been loaded.
 * @returns {Promise<any>} Configured ort instance.
 */
async function ensureOrtLoaded() {
	// If we've already captured the global instance, configure and return it.
	if (ortGlobal) {
		configureOrtEnv(ortGlobal);
		return ortGlobal;
	}

	// If ort is not available here, the script tag is missing or mis-ordered.
	throw new Error("onnxruntime-web global 'ort' is not available.");
}

/**
 * Configure runtime environment for onnxruntime-web.
 * @param {any} ortInstance
 */
function configureOrtEnv(ortInstance) {
	try {
		ortInstance.env.wasm.simd = true;
		ortInstance.env.wasm.numThreads = 1;
		ortInstance.env.wasm.proxy = false;
		ortInstance.env.allowLocalModels = false;

		// Ensure wasm binaries are loaded alongside the CDN script.
		ortInstance.env.wasm.wasmPaths =
			"https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
	} catch {
		// Fail soft: ORT will fall back to its defaults if setting env fails.
	}
}

/**
 * Backwards-compatible helper if legacy code calls getRuntime().
 * @returns {Promise<any>}
 */
async function getRuntime() {
	return ensureOrtLoaded();
}

// ----------------------- Model + CRF loading ------------------------------

/**
 * Fetch and parse the CRF parameter JSON.
 * @returns {Promise<any>}
 */
async function getModelCRFJSON() {
	const url = new URL("model.crf.json", SCRIPT_BASE_URL).toString();
	const res = await fetch(url, { cache: "no-cache" });

	if (!res.ok) {
		throw new Error(`Failed to fetch CRF JSON (${res.status} ${res.statusText})`);
	}

	return res.json();
}

/**
 * Fetch the ONNX model as an ArrayBuffer.
 * @returns {Promise<ArrayBuffer>}
 */
async function getModelBuf() {
	const url = new URL("model.onnx", SCRIPT_BASE_URL).toString();
	const res = await fetch(url, { cache: "no-cache" });

	if (!res.ok) {
		throw new Error(`Failed to fetch ONNX model (${res.status} ${res.statusText})`);
	}

	return res.arrayBuffer();
}

/**
 * Initialize and cache the model bundle.
 *
 * - Ensures onnxruntime-web is ready.
 * - Fetches ./model.onnx and ./model.crf.json (relative to this script).
 * - Creates an InferenceSession and prepares CRF parameters.
 *
 * @returns {Promise<any>} Model bundle used by `runInference`.
 */
async function initModel() {
	if (modelPromise) return modelPromise;

	modelPromise = (async () => {
		await ensureOrtLoaded();

		const [modelBuf, crfJSON] = await Promise.all([
			getModelBuf(),
			getModelCRFJSON()
		]);

		model = await loadModel(modelBuf, crfJSON);
		return model;
	})();

	return modelPromise;
}

// ------------------------------- Config -----------------------------------

/**
 * Number of features per line (dimensionality of each feature vector).
 * @type {number}
 */
const NUM_FEATURES = 16;

// --------------------------- Math helpers ---------------------------------

/**
 * Numerically stable softmax.
 * @param {number[]} vec
 * @returns {number[]}
 */
function softmax(vec) {
	const m = Math.max(...vec);
	const exps = vec.map(v => Math.exp(v - m));
	const s = exps.reduce((a, b) => a + b, 0);
	return exps.map(v => v / s);
}

/**
 * Argmax over a numeric array.
 * @param {number[]} vec
 * @returns {number}
 */
function argmax(vec) {
	let imax = 0;
	let vmax = -Infinity;

	for (let i = 0; i < vec.length; i++) {
		if (vec[i] > vmax) {
			vmax = vec[i];
			imax = i;
		}
	}

	return imax;
}

// ------------------------ Tensor preparation ------------------------------

/**
 * Pad or truncate an array of feature rows to length T.
 *
 * @param {number[][]} lines - Array of feature rows.
 * @param {number} T - Target length.
 * @returns {{ padded: number[][], validCount: number }}
 */
function padOrTruncateLines(lines, T) {
	const padded = new Array(T);
	const n = Math.min(lines.length, T);

	for (let i = 0; i < n; i++) {
		const row = lines[i];
		if (!Array.isArray(row) || row.length !== NUM_FEATURES) {
			throw new Error(`Line ${i} must be an array of length ${NUM_FEATURES}`);
		}
		padded[i] = row;
	}

	const zeroRow = Array(NUM_FEATURES).fill(0);
	for (let i = n; i < T; i++) padded[i] = zeroRow.slice();

	return { padded, validCount: n };
}

/**
 * Build a boolean-like padding mask for a batch, where 1 indicates padded tokens.
 *
 * @param {number} B - Batch size.
 * @param {number} T - Sequence length.
 * @param {number[]} validCounts - Valid token counts per sequence.
 * @returns {Uint8Array}
 */
function buildMask(B, T, validCounts) {
	const mask = new Uint8Array(B * T);
	let k = 0;

	for (let b = 0; b < B; b++) {
		const n = validCounts[b];
		for (let t = 0; t < T; t++) {
			mask[k++] = t >= n ? 1 : 0;
		}
	}

	return mask;
}

/**
 * Flatten a batched [B][T][NUM_FEATURES] array to a contiguous Float32Array.
 *
 * @param {number[][][]} batched
 * @param {number} B
 * @param {number} T
 * @returns {Float32Array}
 */
function flattenFeatures(batched, B, T) {
	const a = new Float32Array(B * T * NUM_FEATURES);
	let k = 0;

	for (let b = 0; b < B; b++) {
		for (let t = 0; t < T; t++) {
			const row = batched[b][t];
			for (let j = 0; j < NUM_FEATURES; j++) {
				a[k++] = row[j];
			}
		}
	}

	return a;
}

// -------------------- CRF runtime (Viterbi decoding) ----------------------

/**
 * Batched Viterbi decoding.
 *
 * @param {Float32Array} emissions - length B*T*C, layout [b, t, c].
 * @param {number} B - Batch size.
 * @param {number} T - Max sequence length.
 * @param {number} C - Number of classes.
 * @param {number[]} validCounts - Valid sequence lengths per batch element.
 * @param {{ transitions: Float32Array, start: Float32Array, end: Float32Array }} crfParams
 * @returns {number[][]} paths[b][t] = class index.
 */
function viterbiDecodeBatch(emissions, B, T, C, validCounts, crfParams) {
	const { transitions, start, end } = crfParams;

	const paths = new Array(B);
	const score = new Float32Array(C);
	const prevScore = new Float32Array(C);

	for (let b = 0; b < B; b++) {
		const n = validCounts[b];
		if (n <= 0) {
			paths[b] = [];
			continue;
		}

		const backpointers = new Int16Array(n * C);

		// t = 0
		{
			const base0 = (b * T + 0) * C;
			for (let j = 0; j < C; j++) {
				score[j] = start[j] + emissions[base0 + j];
				backpointers[j] = 0; // unused at t=0
			}
		}

		// t = 1..n-1
		for (let t = 1; t < n; t++) {
			for (let j = 0; j < C; j++) prevScore[j] = score[j];

			const base = (b * T + t) * C;

			for (let j = 0; j < C; j++) {
				let bestI = 0;
				let bestVal = prevScore[0] + transitions[0 * C + j];

				for (let i = 1; i < C; i++) {
					const val = prevScore[i] + transitions[i * C + j];
					if (val > bestVal) {
						bestVal = val;
						bestI = i;
					}
				}

				score[j] = bestVal + emissions[base + j];
				backpointers[t * C + j] = bestI;
			}
		}

		// end
		let bestLast = 0;
		let bestLastScore = score[0] + end[0];

		for (let j = 1; j < C; j++) {
			const v = score[j] + end[j];
			if (v > bestLastScore) {
				bestLastScore = v;
				bestLast = j;
			}
		}

		// backtrack
		const seq = new Array(n);
		seq[n - 1] = bestLast;

		for (let t = n - 1; t >= 1; t--) {
			seq[t - 1] = backpointers[t * C + seq[t]];
		}

		paths[b] = seq;
	}

	return paths;
}

/**
 * Normalize CRF parameter arrays into flat Float32Array with validation.
 *
 * Accepts:
 * - TypedArray (e.g. Float32Array)
 * - flat number[]
 * - 2D number[][] (flattened row-major)
 *
 * @param {any} arr
 * @param {number | null} expectedLength
 * @param {string} name
 * @returns {Float32Array}
 */
function ensureFloat32Flat(arr, expectedLength, name) {
	if (arr == null) {
		throw new Error(`CRF ${name} is missing`);
	}

	// TypedArray or other views
	if (ArrayBuffer.isView(arr) && typeof arr.length === "number") {
		const out = new Float32Array(arr);
		if (expectedLength != null && out.length !== expectedLength) {
			throw new Error(`CRF ${name} has length ${out.length}, expected ${expectedLength}`);
		}
		return out;
	}

	if (!Array.isArray(arr)) {
		throw new Error(`CRF ${name} must be an array`);
	}

	const first = arr[0];

	// 2D array
	if (Array.isArray(first)) {
		const rows = arr.length;
		const cols = first.length;

		for (let r = 0; r < rows; r++) {
			if (!Array.isArray(arr[r]) || arr[r].length !== cols) {
				throw new Error(
					`CRF ${name} must be a rectangular 2D array [${rows}][${cols}]`
				);
			}
		}

		const flat = new Float32Array(rows * cols);
		let k = 0;

		for (let r = 0; r < rows; r++) {
			for (let c = 0; c < cols; c++) {
				const v = arr[r][c];
				flat[k++] = Number.isFinite(v) ? v : 0;
			}
		}

		if (expectedLength != null && flat.length !== expectedLength) {
			throw new Error(
				`CRF ${name} has flattened length ${flat.length}, expected ${expectedLength}`
			);
		}

		return flat;
	}

	// flat 1D array
	const out = new Float32Array(arr.length);
	for (let i = 0; i < arr.length; i++) {
		const v = arr[i];
		out[i] = Number.isFinite(v) ? v : 0;
	}

	if (expectedLength != null && out.length !== expectedLength) {
		throw new Error(`CRF ${name} has length ${out.length}, expected ${expectedLength}`);
	}

	return out;
}

/**
 * Load an ONNX model and CRF parameters from already-fetched data.
 *
 * @param {ArrayBuffer} modelBuf - ONNX model bytes.
 * @param {Object} crfData - Parsed CRF JSON data.
 * @returns {Promise<{
 *   session: any,
 *   io: { featureInputName: string, maskInputName: string, outputName: string },
 *   crf: { numTags: number, transitions: Float32Array, start: Float32Array, end: Float32Array }
 * }>}
 */
async function loadModel(modelBuf, crfData) {
	await ensureOrtLoaded();

	const session = await ortGlobal.InferenceSession.create(modelBuf, {
		executionProviders: ["wasm"],
		graphOptimizationLevel: "all"
	});

	// Derive I/O names from model metadata with safe fallbacks.
	const inputs = session.inputNames || [];
	const outputs = session.outputNames || [];
	const inMeta = session.inputMetadata || {};

	let featureInputName = "line_features";
	let maskInputName = "pad_mask";
	let outputName = outputs[0] || "emissions";

	const byRank3 = inputs.find(
		n => inMeta?.[n]?.dimensions?.length === 3
	);
	const byRank2 = inputs.find(
		n => inMeta?.[n]?.dimensions?.length === 2
	);

	if (byRank3) featureInputName = byRank3;
	if (byRank2) maskInputName = byRank2;
	if (outputs.length > 0) outputName = outputs[0];

	// Load CRF params JSON
	const j = crfData;
	const numTags = Number(j.num_tags ?? j.numTags);

	if (!Number.isInteger(numTags) || numTags <= 0) {
		throw new Error("CRF JSON missing valid num_tags/numTags");
	}

	const transitions = ensureFloat32Flat(
		j.transitions,
		numTags * numTags,
		"transitions"
	);
	const start = ensureFloat32Flat(
		j.start_transitions ?? j.start,
		numTags,
		"start_transitions"
	);
	const end = ensureFloat32Flat(
		j.end_transitions ?? j.end,
		numTags,
		"end_transitions"
	);

	const crf = { numTags, transitions, start, end };

	return {
		session,
		io: { featureInputName, maskInputName, outputName },
		crf
	};
}

/**
 * Run model inference on a batch of records.
 *
 * @param {Object} params
 * @param {any} params.session - ONNX Runtime session.
 * @param {{ featureInputName: string, maskInputName: string, outputName: string }} params.io
 * @param {{ lines: number[][] }[]} params.records
 * @param {{ T: number }} params.shape
 * @param {{ numTags: number, transitions: Float32Array, start: Float32Array, end: Float32Array } | null} [params.crf]
 *
 * @returns {Promise<{
 *   predictions: Array<Array<{
 *     t: number,
 *     class: number,
 *     type: number,
 *     confidence: number
 *   }>>,
 *   logitsShape: [number, number, number]
 * }>}
 */
async function runInference({ session, io, records, shape: { T }, crf = null }) {
	if (!session) throw new Error("Session is required.");
	if (!io) throw new Error("I/O names are required.");
	if (!Array.isArray(records) || records.length === 0) {
		throw new Error("records must be a non-empty array.");
	}
	if (!Number.isInteger(T) || T <= 0) {
		throw new Error("shape.T must be a positive integer.");
	}

	const B = records.length;

	// Build batch
	const batched = new Array(B);
	const validCounts = new Array(B);

	for (let b = 0; b < B; b++) {
		const rec = records[b];
		if (!rec || !Array.isArray(rec.lines)) {
			throw new Error(`records[${b}] must contain a 'lines' array.`);
		}

		const { padded, validCount } = padOrTruncateLines(rec.lines, T);
		batched[b] = padded;
		validCounts[b] = validCount;
	}

	const features = flattenFeatures(batched, B, T);
	const padMask = buildMask(B, T, validCounts); // 1 for padded, 0 for valid

	const feeds = {};
	feeds[io.featureInputName] = new ort.Tensor(
		"float32",
		features,
		[B, T, NUM_FEATURES]
	);
	feeds[io.maskInputName] = new ort.Tensor(
		"bool",
		padMask,
		[B, T]
	);

	const t0 = performance.now();
	const results = await session.run(feeds);
	const dt = performance.now() - t0;

	// Optional debug: uncomment for timing logs
	// console.log(`Inference took ${dt.toFixed(2)} ms`);

	const out = results[io.outputName];
	if (!out) {
		throw new Error(`Output ${io.outputName} not found.`);
	}

	const dims = out.dims; // expected [B, T, C]
	if (dims.length !== 3 || dims[0] !== B || dims[1] !== T) {
		throw new Error(
			`Unexpected output shape ${dims.join("x")} (expected ${B}x${T}xC).`
		);
	}

	const C = dims[2];
	const emissions = out.data; // Float32Array length B*T*C

	let decodedPaths = null;
	if (crf) {
		if (crf.numTags !== C) {
			throw new Error(`CRF numTags (${crf.numTags}) != emissions classes (${C})`);
		}

		decodedPaths = viterbiDecodeBatch(
			emissions,
			B,
			T,
			C,
			validCounts,
			{
				transitions: crf.transitions,
				start: crf.start,
				end: crf.end
			}
		);
	}

	// Convert to predictions on valid tokens only
	const predictions = [];
	let k = 0; // index into emissions for non-CRF path

	for (let b = 0; b < B; b++) {
		const seq = [];
		const n = validCounts[b];

		if (crf && decodedPaths) {
			// Use CRF-decoded classes
			const path = decodedPaths[b];

			for (let t = 0; t < n; t++) {
				const v = new Array(C);
				for (let c = 0; c < C; c++) {
					v[c] = emissions[(b * T + t) * C + c];
				}

				const probs = softmax(v);
				const cls = path[t];

				seq.push({
					t,
					class: cls,
					type: cls, // `type` is the raw class index
					confidence: Number((probs[cls] ?? 0).toFixed(4))
				});
			}

			// Advance k to skip entire sequence (we used direct indexing above)
			k += T * C;
		} else {
			// Fallback: per-token argmax on emissions (no CRF)
			for (let t = 0; t < n; t++) {
				const v = new Array(C);
				for (let c = 0; c < C; c++) {
					v[c] = emissions[k++];
				}

				const probs = softmax(v);
				const cls = argmax(v);

				seq.push({
					t,
					class: cls,
					type: cls, // `type` is the raw class index
					confidence: Number(probs[cls].toFixed(4))
				});
			}

			// Skip remaining padded tokens' emissions
			k += (T - n) * C;
		}

		predictions.push(seq);
	}

	return {
		predictions,
		logitsShape: [B, T, C]
	};
}

// --------------------- Block post-processing ------------------------------

/**
 * Build logical text blocks from per-line predictions.
 *
 * @param {string[]} lines - Raw text lines.
 * @param {{ type: number }[]} results - Per-line prediction objects.
 * @returns {Array<{ type: string, lines: number[], text: string }>}
 */
function buildBlocks(lines, results) {
	const BLOCK_TYPES = ["title", "other", "body", "list_item", "equation", "frame"];

	if (
		!Array.isArray(lines) ||
		!Array.isArray(results) ||
		lines.length !== results.length
	) {
		throw new Error("lines and results must be arrays of equal length");
	}

	const blocks = [];
	let currentBlock = null;

	const getBaseType = lineType => (lineType >= 6 ? lineType - 6 : lineType);

	for (let i = 0; i < lines.length; i++) {
		const res = results[i];
		const lineType = res?.type ?? 0;
		const baseType = getBaseType(lineType);
		const isFirstLine = lineType <= 5;

		const startNewBlock = () => {
			if (currentBlock) blocks.push(currentBlock);
			currentBlock = {
				// Temporary numeric type; converted to string after grouping.
				type: baseType,
				// Store indices of lines that belong to this block.
				lines: [i]
			};
		};

		if (!currentBlock) {
			// No current block yet: start one.
			startNewBlock();
		} else {
			// Decide whether to continue current block or start a new one.
			if (isFirstLine || currentBlock.type !== baseType) {
				startNewBlock();
			} else {
				// Continue current block.
				currentBlock.lines.push(i);
			}
		}
	}

	if (currentBlock) blocks.push(currentBlock);

	// Map numeric types to string labels and build block text.
	for (const block of blocks) {
		block.type = BLOCK_TYPES[block.type] ?? "other";
		block.text = block.lines
						   .map(idx => lines[idx] ?? "")
						   .join(" ");
	}

	return blocks;
}

// --------------------------- Public API -----------------------------------

/**
 * Run the layout model on a page and return logical text blocks inferred
 * from the lines.
 *
 * @param {number[][]} vectors - Feature vectors, one per line (length NUM_FEATURES each).
 * @param {string[]} lines - Corresponding text lines.
 * @returns {Promise<Array<{ type: string, lines: number[], text: string }>>}
 */
async function inference(vectors, lines) {
	if (!Array.isArray(vectors) || !Array.isArray(lines) || vectors.length !== lines.length) {
		throw new Error("vectors and lines must be arrays of equal length");
	}

	const modelBundle = await initModel();

	const timerName = "Page inference time";
	console.time(timerName);

	const { predictions } = await runInference({
		...modelBundle,
		records: [{ lines: vectors }],
		shape: { T: vectors.length }
	});

	console.timeEnd(timerName);

	const result = predictions[0];

	const blocks = buildBlocks(lines, result);
	return blocks;
}

// Expose the API globally.
window.inference = inference;