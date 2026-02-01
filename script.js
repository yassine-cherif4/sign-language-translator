const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
const currentLetterElement = document.getElementById('current-letter');
const confidenceLevelElement = document.getElementById('confidence-level');
const transcribedTextElement = document.getElementById('transcribed-text');
const statusText = document.getElementById('status-text');
const statusDot = document.querySelector('.status-dot');

// UI State
let lastDetectedLetter = null;
let detectionStartTime = 0;
const DETECTION_THRESHOLD_MS = 600; // Time to hold a sign to type it
let sentence = "";

// Finger State Enums
const Finger = {
    Thumb: 0,
    Index: 1,
    Middle: 2,
    Ring: 3,
    Pinky: 4
};

const FingerCurl = {
    NoCurl: 0,
    HalfCurl: 1,
    FullCurl: 2
};

const FingerDirection = {
    VerticalUp: 0,
    VerticalDown: 1,
    HorizontalLeft: 2,
    HorizontalRight: 3,
    DiagonalUpRight: 4,
    DiagonalUpLeft: 5,
    DiagonalDownRight: 6,
    DiagonalDownLeft: 7
};

class GestureEstimator {
    constructor() {
        this.gestures = [];
    }

    addGesture(name, rules) {
        this.gestures.push({ name, rules });
    }

    estimate(landmarks) {
        // Calculate physics/geometry of hand
        const curls = this.calculateCurls(landmarks);
        // Note: Directions are harder without comprehensive 3D normalization, 
        // using simple relative positions for now.

        let possibleGestures = [];

        for (const gesture of this.gestures) {
            let score = 0;
            let match = true;

            for (const [finger, requiredCurl] of Object.entries(gesture.rules)) {
                if (Array.isArray(requiredCurl)) {
                    if (!requiredCurl.includes(curls[finger])) {
                        match = false;
                        break;
                    }
                } else {
                    if (curls[finger] !== requiredCurl) {
                        match = false;
                        break;
                    }
                }
                score++;
            }

            if (match) {
                possibleGestures.push({ name: gesture.name, score: score });
            }
        }

        // Sort by score (specificity)
        possibleGestures.sort((a, b) => b.score - a.score);
        return possibleGestures.length > 0 ? possibleGestures[0] : null;
    }

    calculateCurls(landmarks) {
        const curls = {};
        const wrist = landmarks[0];
        const fingerTips = [4, 8, 12, 16, 20];
        const fingerPip = [2, 6, 10, 14, 18];
        const fingerMcp = [1, 5, 9, 13, 17];

        // Thumb: Check if tip is "across" palm (towards pinky)
        // Improved: simple localized distance check.
        // If thumb tip is closer to pinky MCP than thumb MCP is.
        const thumbTip = landmarks[4];
        const thumbMcp = landmarks[2];
        const pinkyMcp = landmarks[17];

        const mcpDist = this.getDistance(thumbMcp, pinkyMcp);
        const tipDist = this.getDistance(thumbTip, pinkyMcp);

        // Tuned threshold: tip must be significantly closer
        curls[Finger.Thumb] = (tipDist < mcpDist * 0.9) ? FingerCurl.FullCurl : FingerCurl.NoCurl;

        // Fingers - Improved HalfCurl support
        for (let f = 1; f < 5; f++) {
            const tip = landmarks[fingerTips[f]];
            const mcp = landmarks[fingerMcp[f]];

            const tipToWrist = this.getDistance(tip, wrist);
            const mcpToWrist = this.getDistance(mcp, wrist);
            const ratio = tipToWrist / mcpToWrist;

            // Tuned thresholds for Curl states
            if (ratio < 0.85) {
                curls[f] = FingerCurl.FullCurl;
            } else if (ratio < 1.3) {
                curls[f] = FingerCurl.HalfCurl;
            } else {
                curls[f] = FingerCurl.NoCurl;
            }
        }

        return curls;
    }

    getDistance(p1, p2) {
        return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
    }
}

class TrajectoryTracker {
    constructor(maxSize = 25) {
        this.points = [];
        this.maxSize = maxSize;
    }

    add(point) {
        this.points.push({ ...point, t: Date.now() });
        if (this.points.length > this.maxSize) this.points.shift();
    }

    clear() {
        this.points = [];
    }

    detectHook() { // For J
        if (this.points.length < 10) return false;

        // J is basically a 'U' shape or hook shape in Y-axis
        const start = this.points[0];
        const minHeight = Math.min(...this.points.map(p => p.y));
        const maxHeight = Math.max(...this.points.map(p => p.y));
        const last = this.points[this.points.length - 1];

        // Need significant downward then upward movement
        const depth = maxHeight - minHeight;
        const recovered = maxHeight - last.y;

        // Looking for: Down, then curve up
        return depth > 0.1 && recovered > depth * 0.4;
    }

    detectZigZag() { // For Z
        if (this.points.length < 15) return false;

        // Trace X movement over time
        let directionalChanges = 0;
        let lastDir = 0; // 1 for right, -1 for left

        for (let i = 1; i < this.points.length; i++) {
            const dx = this.points[i].x - this.points[i - 1].x;
            if (Math.abs(dx) > 0.005) {
                const dir = dx > 0 ? 1 : -1;
                if (lastDir !== 0 && dir !== lastDir) {
                    directionalChanges++;
                }
                lastDir = dir;
            }
        }

        // Z has 2 major X-direction changes: Right -> Left -> Right
        return directionalChanges >= 2;
    }

    detectWave() {
        if (this.points.length < 15) return false;

        // Trace X movement over time - WAVE is higher frequency/count than Z
        let directionalChanges = 0;
        let lastDir = 0;

        for (let i = 1; i < this.points.length; i++) {
            const dx = this.points[i].x - this.points[i - 1].x;
            if (Math.abs(dx) > 0.01) {
                const dir = dx > 0 ? 1 : -1;
                if (lastDir !== 0 && dir !== lastDir) {
                    directionalChanges++;
                }
                lastDir = dir;
            }
        }

        // Wave has 3+ major X-direction changes
        return directionalChanges >= 3;
    }
}

// Setup Recognizer
const estimator = new GestureEstimator();
const indexTracker = new TrajectoryTracker();
const pinkyTracker = new TrajectoryTracker();

// Define Rules (Improved Heuristics)
// Permissive Thumb strategies to improve detection rate.

// A: Fingers Curled. Thumb Up/Side (NoCurl).
estimator.addGesture('A', {
    [Finger.Thumb]: FingerCurl.NoCurl,
    [Finger.Index]: FingerCurl.FullCurl,
    [Finger.Middle]: FingerCurl.FullCurl,
    [Finger.Ring]: FingerCurl.FullCurl,
    [Finger.Pinky]: FingerCurl.FullCurl
});

// B: Fingers Open. Thumb Tucked (FullCurl) or HalfCurl. 
// Removed NoCurl to prevent shadowing 'C' or '5'.
estimator.addGesture('B', {
    [Finger.Thumb]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
    [Finger.Index]: FingerCurl.NoCurl,
    [Finger.Middle]: FingerCurl.NoCurl,
    [Finger.Ring]: FingerCurl.NoCurl,
    [Finger.Pinky]: FingerCurl.NoCurl
});

// C: All Open (approximated). 
// Since we removed NoCurl from B, C will now be the match for "Open Hand" states.
estimator.addGesture('C', {
    [Finger.Thumb]: [FingerCurl.NoCurl, FingerCurl.HalfCurl],
    [Finger.Index]: [FingerCurl.NoCurl, FingerCurl.HalfCurl],
    [Finger.Middle]: [FingerCurl.NoCurl, FingerCurl.HalfCurl],
    [Finger.Ring]: [FingerCurl.NoCurl, FingerCurl.HalfCurl],
    [Finger.Pinky]: [FingerCurl.NoCurl, FingerCurl.HalfCurl]
});


// D: Index Up. Others Curled. Thumb touching middle (Curled).
// Removed NoCurl to prevent shadowing 'L'.
estimator.addGesture('D', {
    [Finger.Thumb]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
    [Finger.Index]: FingerCurl.NoCurl,
    [Finger.Middle]: FingerCurl.FullCurl,
    [Finger.Ring]: FingerCurl.FullCurl,
    [Finger.Pinky]: FingerCurl.FullCurl
});

// F: Index & Thumb Curled (Touch). Others Open.
estimator.addGesture('F', {
    [Finger.Thumb]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
    [Finger.Index]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
    [Finger.Middle]: FingerCurl.NoCurl,
    [Finger.Ring]: FingerCurl.NoCurl,
    [Finger.Pinky]: FingerCurl.NoCurl
});

// L: Thumb/Index Open. Others Curled.
estimator.addGesture('L', {
    [Finger.Thumb]: FingerCurl.NoCurl,
    [Finger.Index]: FingerCurl.NoCurl,
    [Finger.Middle]: FingerCurl.FullCurl,
    [Finger.Ring]: FingerCurl.FullCurl,
    [Finger.Pinky]: FingerCurl.FullCurl
});

// S: Fist. All Curled.
estimator.addGesture('S', {
    [Finger.Thumb]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
    [Finger.Index]: FingerCurl.FullCurl,
    [Finger.Middle]: FingerCurl.FullCurl,
    [Finger.Ring]: FingerCurl.FullCurl,
    [Finger.Pinky]: FingerCurl.FullCurl
});

// E: All fingers Curled. Thumb Curled (Tucked or side).
// Often conflicts with S. E usually has thumb lower or touching tips?
// We'll define it broadly and maybe use a specialized check if needed.
estimator.addGesture('E', {
    [Finger.Thumb]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
    [Finger.Index]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
    [Finger.Middle]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
    [Finger.Ring]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
    [Finger.Pinky]: [FingerCurl.FullCurl, FingerCurl.HalfCurl]
});

// G: Index Open. Others Curled. (Horizontal).
// Defined same as D, resolved by Orientation Check.
estimator.addGesture('G', {
    [Finger.Thumb]: [FingerCurl.NoCurl, FingerCurl.HalfCurl, FingerCurl.FullCurl],
    [Finger.Index]: [FingerCurl.NoCurl, FingerCurl.HalfCurl], // Allow HalfCurl for foreshortening
    [Finger.Middle]: FingerCurl.FullCurl,
    [Finger.Ring]: FingerCurl.FullCurl,
    [Finger.Pinky]: FingerCurl.FullCurl
});

// H: Index/Middle Open. Others Curled. (Horizontal).
// Defined same as U/V, resolved by Orientation Check.
estimator.addGesture('H', {
    [Finger.Thumb]: [FingerCurl.NoCurl, FingerCurl.HalfCurl, FingerCurl.FullCurl],
    [Finger.Index]: FingerCurl.NoCurl,
    [Finger.Middle]: FingerCurl.NoCurl,
    [Finger.Ring]: FingerCurl.FullCurl,
    [Finger.Pinky]: FingerCurl.FullCurl
});

// R: Index/Middle Open & Crossed.
// Defined same as U, resolved by "Cross" Check.
estimator.addGesture('R', {
    [Finger.Thumb]: [FingerCurl.NoCurl, FingerCurl.HalfCurl, FingerCurl.FullCurl],
    [Finger.Index]: FingerCurl.NoCurl,
    [Finger.Middle]: FingerCurl.NoCurl,
    [Finger.Ring]: FingerCurl.FullCurl,
    [Finger.Pinky]: FingerCurl.FullCurl
});

// X: Index Curled (Hook/Half). Others Curled.
// Relaxed to allow HalfCurl on support fingers (loose fist).
estimator.addGesture('X', {
    [Finger.Thumb]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
    [Finger.Index]: [FingerCurl.HalfCurl, FingerCurl.FullCurl], // Allow Full too if hook is tight
    [Finger.Middle]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
    [Finger.Ring]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
    [Finger.Pinky]: [FingerCurl.FullCurl, FingerCurl.HalfCurl]
});

// K: Index/Middle Open. Thumb Open (Straight up).
estimator.addGesture('K', {
    [Finger.Thumb]: FingerCurl.NoCurl,
    [Finger.Index]: FingerCurl.NoCurl,
    [Finger.Middle]: FingerCurl.NoCurl,
    [Finger.Ring]: FingerCurl.FullCurl,
    [Finger.Pinky]: FingerCurl.FullCurl
});

// O: All fingers Curled (or Half). Thumb Open/Half (Touching index).
// Distinction from S: O has thumb somewhat open?
// Allow NoCurl to catch "loose O" which looks like C to the curl estimator.
estimator.addGesture('O', {
    [Finger.Thumb]: [FingerCurl.NoCurl, FingerCurl.HalfCurl],
    [Finger.Index]: [FingerCurl.FullCurl, FingerCurl.HalfCurl, FingerCurl.NoCurl],
    [Finger.Middle]: [FingerCurl.FullCurl, FingerCurl.HalfCurl, FingerCurl.NoCurl],
    [Finger.Ring]: [FingerCurl.FullCurl, FingerCurl.HalfCurl, FingerCurl.NoCurl],
    [Finger.Pinky]: [FingerCurl.FullCurl, FingerCurl.HalfCurl, FingerCurl.NoCurl]
});

// U: Index/Middle Open. Thumb Tucked. 
// (Same curls as V, but visually fingers merged. Code can't distinguish yet, but added for completeness).
estimator.addGesture('U', {
    [Finger.Thumb]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
    [Finger.Index]: FingerCurl.NoCurl,
    [Finger.Middle]: FingerCurl.NoCurl,
    [Finger.Ring]: FingerCurl.FullCurl,
    [Finger.Pinky]: FingerCurl.FullCurl
});

// V: Index/Middle Open. Thumb Tucked.
// Removed NoCurl from Thumb to allow K (which has Thumb NoCurl).
estimator.addGesture('V', {
    [Finger.Thumb]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
    [Finger.Index]: FingerCurl.NoCurl,
    [Finger.Middle]: FingerCurl.NoCurl,
    [Finger.Ring]: FingerCurl.FullCurl,
    [Finger.Pinky]: FingerCurl.FullCurl
});

// I: Pinky Open. Others Curled.
estimator.addGesture('I', {
    [Finger.Thumb]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
    [Finger.Index]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
    [Finger.Middle]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
    [Finger.Ring]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
    [Finger.Pinky]: FingerCurl.NoCurl
});

// W: Index/Middle/Ring Open. Pinky Curled.
estimator.addGesture('W', {
    [Finger.Thumb]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
    [Finger.Index]: FingerCurl.NoCurl,
    [Finger.Middle]: FingerCurl.NoCurl,
    [Finger.Ring]: FingerCurl.NoCurl,
    [Finger.Pinky]: FingerCurl.FullCurl
});

// Y: Thumb & Pinky open.
estimator.addGesture('Y', {
    [Finger.Thumb]: [FingerCurl.NoCurl, FingerCurl.HalfCurl], // Relaxed to allow HalfCurl
    [Finger.Index]: FingerCurl.FullCurl,
    [Finger.Middle]: FingerCurl.FullCurl,
    [Finger.Ring]: FingerCurl.FullCurl,
    [Finger.Pinky]: [FingerCurl.NoCurl, FingerCurl.HalfCurl] // Relaxed
});

// M, N, T (Fist Group Variants)
// Defined same as S/E, resolved by Thumb-MCP position.
['M', 'N', 'T'].forEach(name => {
    estimator.addGesture(name, {
        [Finger.Thumb]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
        [Finger.Index]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
        [Finger.Middle]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
        [Finger.Ring]: [FingerCurl.FullCurl, FingerCurl.HalfCurl],
        [Finger.Pinky]: [FingerCurl.FullCurl, FingerCurl.HalfCurl]
    });
});

// P and Q removed from base estimator to avoid shadowing base shapes K and G.
// They will be mapped via Orientation Check in onResults.


function onResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        statusText.innerText = "Hand Detected";
        statusDot.classList.add('active');

        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS,
                { color: '#00FF00', lineWidth: 1 }); // Reduced from 5
            drawLandmarks(canvasCtx, landmarks, { color: '#FF0000', lineWidth: 1, radius: 2 }); // Reduced/Fixed size

            // Estimate Gesture
            const estimated = estimator.estimate(landmarks);

            // Visual Debugging: Show curl detections on screen
            const curls = estimator.calculateCurls(landmarks);
            canvasCtx.fillStyle = "white";
            canvasCtx.font = "12px monospace";
            canvasCtx.fillText(`Thumb: ${curls[0] === 2 ? 'Curl' : 'Open'}`, 10, 20);
            canvasCtx.fillText(`Index: ${curls[1] === 2 ? 'Curl' : 'Open'}`, 10, 35);
            canvasCtx.fillText(`Mid:   ${curls[2] === 2 ? 'Curl' : 'Open'}`, 10, 50);
            canvasCtx.fillText(`Ring:  ${curls[3] === 2 ? 'Curl' : 'Open'}`, 10, 65);
            canvasCtx.fillText(`Pinky: ${curls[4] === 2 ? 'Curl' : 'Open'}`, 10, 80);

            // Helper for debug
            let debugLog = "";

            // Console log occasionally
            if (Date.now() % 1000 < 50) {
                console.log("Curls:", curls, "Est:", estimated ? estimated.name : "None");
            }

            if (estimated && estimated.score >= 4) {
                let detectedGesture = estimated.name;

                const wrist = landmarks[0];
                const indexTip = landmarks[8];
                const indexMcp = landmarks[5];
                const middleTip = landmarks[12];
                const middleMcp = landmarks[9];
                const thumbTip = landmarks[4];

                // Calculate Shared Geometry
                const handScale = estimator.getDistance(wrist, middleMcp);
                const dx = indexTip.x - indexMcp.x;
                const dy = indexTip.y - indexMcp.y;
                const mdy = middleTip.y - middleMcp.y;
                const pinkyTip = landmarks[20];

                const isHorizontal = Math.abs(dx) > Math.abs(dy) * 1.6;
                const indexDownward = dy > handScale * 0.45;
                const middleDownward = mdy > handScale * 0.45;
                const isDownward = indexDownward || middleDownward;

                const indexTipToWrist = estimator.getDistance(indexTip, wrist);
                const indexMcpToWrist = estimator.getDistance(indexMcp, wrist);
                const idxRatio = indexTipToWrist / indexMcpToWrist;

                // Update trackers
                indexTracker.add(indexTip);
                pinkyTracker.add(pinkyTip);

                // 1. Orientation Check (Vertical vs Horizontal vs Down)
                if (['D', 'L', 'U', 'V', 'R', 'H', 'G', 'X', 'K', 'C', 'I', 'Y'].includes(detectedGesture)) {
                    canvasCtx.fillText(`Orient: ${isDownward ? 'Down' : (isHorizontal ? 'Horiz' : 'Vert')}`, 120, 125);
                    canvasCtx.fillText(`IdxDown: ${indexDownward ? 'Y' : 'N'} (${(dy / handScale).toFixed(2)})`, 120, 140);

                    // MOTION DETECTION (J & Z)
                    if (['I', 'Y'].includes(detectedGesture) && pinkyTracker.detectHook()) {
                        detectedGesture = 'J';
                        pinkyTracker.clear();
                    } else if (detectedGesture === 'D' && indexTracker.detectZigZag()) {
                        detectedGesture = 'Z';
                        indexTracker.clear();
                    } else if (detectedGesture === 'C' && indexTracker.detectWave()) {
                        detectedGesture = 'HELLO';
                        indexTracker.clear();
                    }
                    // Continue with static orientation logic if no motion detected
                    // PRIORITY: Specificity beats order
                    else if (indexDownward && ['G', 'X', 'L', 'C'].includes(detectedGesture)) {
                        detectedGesture = 'Q';
                    } else if (middleDownward && ['K', 'V', 'U', 'R', 'D'].includes(detectedGesture)) {
                        detectedGesture = 'P';
                    } else if (isHorizontal) {
                        if (['D', 'L', 'X'].includes(detectedGesture)) detectedGesture = 'G';
                        if (['U', 'V', 'R'].includes(detectedGesture)) detectedGesture = 'H';
                    } else {
                        // Force vertical variants
                        if (detectedGesture === 'G') {
                            if (idxRatio < 1.20) detectedGesture = 'X';
                            else detectedGesture = 'D';
                        }
                        if (detectedGesture === 'H') detectedGesture = 'U';
                    }
                }

                // --- CONFLICT RESOLUTION PHASE ---

                // 2. K vs V vs U vs R
                if (detectedGesture === 'K' || detectedGesture === 'V' || detectedGesture === 'U' || detectedGesture === 'R') {
                    const mcpOrder = indexMcp.x < middleMcp.x;
                    const tipOrder = indexTip.x < middleTip.x;

                    if (mcpOrder !== tipOrder) {
                        detectedGesture = 'R';
                        canvasCtx.fillText("R: Crossed", 120, 20);
                    } else {
                        // K Check (Thumb Up)
                        const thumbUp = thumbTip.y < indexMcp.y; // Y is smaller upwards
                        canvasCtx.fillText(`KV Thumb: ${thumbUp ? 'Up' : 'Down'} (${thumbTip.y.toFixed(3)})`, 120, 20);

                        if (thumbUp) {
                            detectedGesture = 'K';
                        } else {
                            // U vs V
                            const fingerDistance = estimator.getDistance(indexTip, middleTip);
                            const uvRatio = fingerDistance / handScale;
                            canvasCtx.fillText(`UV Ratio: ${uvRatio.toFixed(2)} `, 120, 35);

                            if (uvRatio < 0.25) {
                                detectedGesture = 'U';
                            } else {
                                detectedGesture = 'V';
                            }
                        }
                    }
                }

                // 3. D vs X
                if (detectedGesture === 'D' || detectedGesture === 'X') {
                    const ratio = indexTipToWrist / indexMcpToWrist;
                    canvasCtx.fillText(`DX Ratio: ${ratio.toFixed(2)} `, 120, 20);

                    // Tuned threshold: 1.12
                    if (ratio > 1.12) {
                        detectedGesture = 'D';
                    } else {
                        detectedGesture = 'X';
                    }
                }

                // 4. B vs F
                if (detectedGesture === 'B' || detectedGesture === 'F') {
                    const thumbIndexDist = estimator.getDistance(thumbTip, indexTip);
                    const touchRatio = thumbIndexDist / handScale;

                    canvasCtx.fillText(`BF Dist: ${touchRatio.toFixed(2)} `, 120, 20);

                    if (touchRatio < 0.28) { // Relaxed slightly from 0.25
                        detectedGesture = 'F';
                    } else {
                        detectedGesture = 'B';
                    }
                }

                // 5. Fist Group (O, C, E, S, A) + X fallback
                if (['O', 'C', 'E', 'S', 'A', 'X', 'M', 'N', 'T'].includes(detectedGesture)) {
                    // Geometric features
                    const middleTipToWrist = estimator.getDistance(middleTip, wrist);
                    const middleMcpToWrist = estimator.getDistance(middleMcp, wrist);
                    const midRatio = middleTipToWrist / middleMcpToWrist;

                    const thumbIndexDist = estimator.getDistance(thumbTip, indexTip);
                    const oRatio = thumbIndexDist / handScale;

                    canvasCtx.fillText(`MidRatio: ${midRatio.toFixed(2)}`, 120, 20);
                    canvasCtx.fillText(`O Dist: ${oRatio.toFixed(2)}`, 120, 35);

                    // C Check: "Open" Fingers or "Open" Gap
                    // Stricter: Don't call it C if the hand is pointing down (likely Q).
                    // Also check for tucked thumb later to distinguish from T.
                    const thumbTipToMidMcp = estimator.getDistance(thumbTip, middleMcp);
                    const sRatioCheck = thumbTipToMidMcp / handScale;

                    if (!isDownward && (midRatio > 1.15 || (oRatio > 0.4 && midRatio > 0.95 && sRatioCheck > 0.6))) {
                        detectedGesture = 'C';
                    } else {
                        // Closed Hand (Fist-ish)

                        // O/E Check: Tips Touch?
                        // Relaxed threshold to 0.32
                        if (oRatio < 0.32) {
                            // Tips are close. Could be O (Circle) or E (Fist).
                            // Check Index compactness.
                            const indexTipToWrist = estimator.getDistance(indexTip, wrist);
                            const indexMcpToWrist = estimator.getDistance(indexMcp, wrist);
                            const indexRatio = indexTipToWrist / indexMcpToWrist;

                            canvasCtx.fillText(`EO IndRatio: ${indexRatio.toFixed(2)} `, 120, 50);

                            // Relaxed to 0.9 to accept "tighter" O's
                            if (indexRatio > 0.90) {
                                detectedGesture = 'O';
                            } else {
                                detectedGesture = 'E'; // Compact Fist (Index curled down)
                            }
                        } else {
                            // Not touching. A, E, S, X, M, N, T.
                            const indexTipToWrist = estimator.getDistance(indexTip, wrist);
                            const indexMcpToWrist = estimator.getDistance(indexMcp, wrist);
                            const indexRatio = indexTipToWrist / indexMcpToWrist;

                            // Unified Not-Touching Logic (S, A, X, E, M, N, T)

                            // 1. S Check (Priority: Crossing)
                            const thumbTipToMidMcp = estimator.getDistance(thumbTip, middleMcp);
                            const sRatio = thumbTipToMidMcp / handScale;
                            canvasCtx.fillText(`S Dist: ${sRatio.toFixed(2)}`, 120, 65);

                            // M, N, T Distinction (Thumb Position)
                            const ringMcp = landmarks[13];
                            const pinkyMcp = landmarks[17];
                            const thumbX = thumbTip.x;

                            // Helper for checking if thumb is between two finger knuckles
                            const isBetween = (val, a, b) => val > Math.min(a, b) && val < Math.max(a, b);

                            if (sRatio < 0.55) {
                                if (isBetween(thumbX, indexMcp.x, middleMcp.x)) detectedGesture = 'T';
                                else if (isBetween(thumbX, middleMcp.x, ringMcp.x)) detectedGesture = 'N';
                                else if (isBetween(thumbX, ringMcp.x, pinkyMcp.x)) detectedGesture = 'M';
                                else {
                                    // S or E (Crossing/Tucked)
                                    // Height Check
                                    if (thumbTip.y > indexMcp.y + (handScale * 0.15)) {
                                        detectedGesture = 'E'; // Low
                                    } else {
                                        detectedGesture = 'S'; // Crossed
                                    }
                                }
                            } else {
                                // 2. A Check (Thumb Up)
                                if (thumbTip.y < indexMcp.y) {
                                    detectedGesture = 'A';
                                } else {
                                    // 3. X Check (Index Hook/Extension)
                                    // Relaxed threshold to 0.88 from 0.98 to make X easier
                                    if (indexRatio > 0.88) {
                                        detectedGesture = 'X';
                                    } else {
                                        // 4. Default E
                                        detectedGesture = 'E';
                                    }
                                }
                            }
                        }
                    }
                }

                // 6. I vs Y
                // I: Pinky Up, Thumb Tucked.
                // Y: Pinky Up, Thumb Out.
                if (detectedGesture === 'I' || detectedGesture === 'Y') {
                    // Check Thumb Extension
                    // Y: Thumb is extended away from hand.
                    // I: Thumb is curled/tucked against index.

                    // Use distance from Thumb Tip to Index MCP
                    // (Index is curled, so MCP is stable ref)
                    const thumbToIndexMcp = estimator.getDistance(thumbTip, indexMcp);
                    const handScale = estimator.getDistance(wrist, middleMcp);
                    const iyRatio = thumbToIndexMcp / handScale;

                    canvasCtx.fillText(`IY Dist: ${iyRatio.toFixed(2)} `, 120, 20);

                    // Threshold: Y is usually > 0.4. I is < 0.3.
                    if (iyRatio > 0.35) {
                        detectedGesture = 'Y';
                    } else {
                        detectedGesture = 'I';
                    }
                }

                // 6. I vs Y logic above...

                // 7. Final Global Rescues
                // Rescue Y from A/E/S if Pinky is extended
                if (detectedGesture === 'A' || detectedGesture === 'E' || detectedGesture === 'S' || detectedGesture === 'X') {
                    const pinkyTip = landmarks[20];
                    const pinkyMcp = landmarks[17];
                    const pinkyTipToWrist = estimator.getDistance(pinkyTip, wrist);
                    const pinkyMcpToWrist = estimator.getDistance(pinkyMcp, wrist);
                    const pinkyRatio = pinkyTipToWrist / pinkyMcpToWrist;

                    canvasCtx.fillText(`PinkyRat: ${pinkyRatio.toFixed(2)} `, 120, 110);

                    if (pinkyRatio > 1.1) {
                        detectedGesture = 'Y';
                    }
                }

                handleDetection(detectedGesture);
            } else {
                handleDetection(null);
            }
        }
    } else {
        statusText.innerText = "Waiting for hand...";
        statusDot.classList.remove('active');
        handleDetection(null);
    }
    canvasCtx.restore();
}

function handleDetection(gestureName) {
    if (gestureName) {
        currentLetterElement.innerText = gestureName;
        confidenceLevelElement.style.width = '100%';

        if (lastDetectedLetter === gestureName) {
            if (Date.now() - detectionStartTime > DETECTION_THRESHOLD_MS) {
                appendLetter(gestureName);
                detectionStartTime = Date.now(); // Reset to prevent rapid fire
                // Clear trackers on success to prevent accidental double-detection
                indexTracker.clear();
                pinkyTracker.clear();
            }
        } else {
            lastDetectedLetter = gestureName;
            detectionStartTime = Date.now();

            // For shortcuts and special letters, force immediate append
            if (gestureName === 'J' || gestureName === 'Z' || gestureName === 'HELLO') {
                if (gestureName === 'HELLO') {
                    appendWord("HELLO");
                } else {
                    appendLetter(gestureName);
                }
                detectionStartTime = Date.now() + 800; // Longer delay for words
                lastDetectedLetter = null; // Reset
            }
        }
    } else {
        currentLetterElement.innerText = "-";
        confidenceLevelElement.style.width = '0%';
        lastDetectedLetter = null;
    }
}

function appendLetter(letter) {
    // If it's the first letter, clear the placeholder
    if (sentence === "") {
        transcribedTextElement.innerHTML = "";
    }

    sentence += letter;
    // visual feedback (animation could be added here)
    const span = document.createElement('span');
    span.innerText = letter === " " ? "\u00A0" : letter; // Non-breaking space for visibility
    transcribedTextElement.appendChild(span);

    // Auto scroll
    transcribedTextElement.scrollTop = transcribedTextElement.scrollHeight;
}

function appendWord(word) {
    if (sentence === "") {
        transcribedTextElement.innerHTML = "";
    } else if (!sentence.endsWith(" ")) {
        appendLetter(" ");
    }

    for (const char of word) {
        appendLetter(char);
    }
    appendLetter(" "); // Add trailing space
}

const hands = new Hands({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }
});

hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

hands.onResults(onResults);

// Camera Setup
const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({ image: videoElement });
    },
    width: 1280,
    height: 720
});

camera.start()
    .then(() => {
        statusText.innerText = "Camera Active";
    })
    .catch(err => {
        console.error(err);
        statusText.innerText = "Camera Error";
    });

let isCameraActive = true;
const cameraBtn = document.getElementById('camera-btn');

cameraBtn.addEventListener('click', () => {
    if (isCameraActive) {
        camera.stop();
        statusText.innerText = "Camera Paused";
        statusDot.classList.remove('active');
        cameraBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M15 12a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h1.172a3 3 0 0 0 2.12-.879l.83-.828A1 1 0 0 1 6.827 3h2.344a1 1 0 0 1 .707.293l.828.828A3 3 0 0 0 12.828 5H14a1 1 0 0 1 1 1v6zM2 4a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2h-1.172a2 2 0 0 1-1.414-.586l-.828-.828A2 2 0 0 0 9.172 2H6.828a2 2 0 0 0-1.414.586l-.828.828A2 2 0 0 1 3.172 4H2z"/><path d="M8 11a2.5 2.5 0 1 1 0-5 2.5 2.5 0 0 1 0 5zm0 1a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7zM3 6.5a.5.5 0 1 1-1 0 .5.5 0 0 1 1 0z"/></svg>
            Start Camera
        `;
        // Clear canvas
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        isCameraActive = false;
    } else {
        camera.start()
            .then(() => {
                statusText.innerText = "Camera Active";
                cameraBtn.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M15 12a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h1.172a3 3 0 0 0 2.12-.879l.83-.828A1 1 0 0 1 6.827 3h2.344a1 1 0 0 1 .707.293l.828.828A3 3 0 0 0 12.828 5H14a1 1 0 0 1 1 1v6zM2 4a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2h-1.172a2 2 0 0 1-1.414-.586l-.828-.828A2 2 0 0 0 9.172 2H6.828a2 2 0 0 0-1.414.586l-.828.828A2 2 0 0 1 3.172 4H2z"/><path d="M8 11a2.5 2.5 0 1 1 0-5 2.5 2.5 0 0 1 0 5zm0 1a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7zM3 6.5a.5.5 0 1 1-1 0 .5.5 0 0 1 1 0z"/></svg>
                Stop Camera
            `;
                isCameraActive = true;
            })
            .catch(err => {
                console.error("Error restarting camera:", err);
                statusText.innerText = "Error";
            });
    }
});


// Buttons
document.getElementById('space-btn').addEventListener('click', () => {
    appendLetter(" ");
});

document.getElementById('delete-btn').addEventListener('click', () => {
    if (sentence.length > 0) {
        sentence = sentence.slice(0, -1);
        if (sentence === "") {
            transcribedTextElement.innerHTML = '<span class="placeholder">Start signing to translate...</span>';
        } else {
            // Re-render the spans for performance/stability or just remove last child
            if (transcribedTextElement.lastElementChild) {
                transcribedTextElement.removeChild(transcribedTextElement.lastElementChild);
            }
        }
    }
});

document.getElementById('clear-btn').addEventListener('click', () => {
    sentence = "";
    transcribedTextElement.innerHTML = '<span class="placeholder">Start signing to translate...</span>';
});

document.getElementById('speak-btn').addEventListener('click', () => {
    if (sentence.length > 0) {
        const utterance = new SpeechSynthesisUtterance(sentence);
        window.speechSynthesis.speak(utterance);
    }
});
