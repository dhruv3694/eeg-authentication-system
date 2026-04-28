export function highpass(values, cutoff, fs) {
    if (!values || values.length === 0 || cutoff <= 0) return values;
    const rc = 1.0 / (2.0 * Math.PI * cutoff);
    const dt = 1.0 / fs;
    const alpha = rc / (rc + dt);
    const out = new Float32Array(values.length);
    let prev_y = 0.0;
    let prev_x = values[0];
    for (let i = 0; i < values.length; i++) {
        const x = values[i];
        const y = alpha * (prev_y + x - prev_x);
        out[i] = y;
        prev_y = y;
        prev_x = x;
    }
    return out;
}

export function lowpass(values, cutoff, fs) {
    if (!values || values.length === 0 || cutoff <= 0) return values;
    const rc = 1.0 / (2.0 * Math.PI * cutoff);
    const dt = 1.0 / fs;
    const alpha = dt / (rc + dt);
    const out = new Float32Array(values.length);
    let y = values[0];
    for (let i = 0; i < values.length; i++) {
        const x = values[i];
        y += alpha * (x - y);
        out[i] = y;
    }
    return out;
}

export function notch(values, freq, fs, q = 30.0) {
    if (!values || values.length < 3 || freq <= 0 || freq >= fs / 2) return values;
    const w0 = 2.0 * Math.PI * freq / fs;
    const alpha = Math.sin(w0) / (2.0 * q);
    const cos_w0 = Math.cos(w0);
    let b0 = 1.0;
    let b1 = -2.0 * cos_w0;
    let b2 = 1.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    b0 /= a0;
    b1 /= a0;
    b2 /= a0;
    a1 /= a0;
    a2 /= a0;

    const out = new Float32Array(values.length);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < values.length; i++) {
        const x0 = values[i];
        const y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        out[i] = y0;
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
    }
    return out;
}

export const MONTAGES = {
    "Longitudinal bipolar": [
        ["Fp1-F7", "Fp1", "F7"], ["F7-T3", "F7", "T3"], ["T3-T5", "T3", "T5"], ["T5-O1", "T5", "O1"],
        ["Fp1-F3", "Fp1", "F3"], ["F3-C3", "F3", "C3"], ["C3-P3", "C3", "P3"], ["P3-O1", "P3", "O1"],
        ["Fp2-F8", "Fp2", "F8"], ["F8-T4", "F8", "T4"], ["T4-T6", "T4", "T6"], ["T6-O2", "T6", "O2"],
        ["Fp2-F4", "Fp2", "F4"], ["F4-C4", "F4", "C4"], ["C4-P4", "C4", "P4"], ["P4-O2", "P4", "O2"],
        ["Fz-Cz", "Fz", "Cz"], ["Cz-Pz", "Cz", "Pz"],
    ],
    "Transverse bipolar": [
        ["Fp1-Fp2", "Fp1", "Fp2"], ["F7-F3", "F7", "F3"], ["F3-Fz", "F3", "Fz"], ["Fz-F4", "Fz", "F4"], ["F4-F8", "F4", "F8"],
        ["T3-C3", "T3", "C3"], ["C3-Cz", "C3", "Cz"], ["Cz-C4", "Cz", "C4"], ["C4-T4", "C4", "T4"],
        ["T5-P3", "T5", "P3"], ["P3-Pz", "P3", "Pz"], ["Pz-P4", "Pz", "P4"], ["P4-T6", "P4", "T6"],
        ["O1-O2", "O1", "O2"],
    ],
    "Average reference": [
        ["Fp1-AVG", "Fp1", "AVG"], ["Fp2-AVG", "Fp2", "AVG"], ["F7-AVG", "F7", "AVG"], ["F3-AVG", "F3", "AVG"],
        ["Fz-AVG", "Fz", "AVG"], ["F4-AVG", "F4", "AVG"], ["F8-AVG", "F8", "AVG"], ["T3-AVG", "T3", "AVG"],
        ["C3-AVG", "C3", "AVG"], ["Cz-AVG", "Cz", "AVG"], ["C4-AVG", "C4", "AVG"], ["T4-AVG", "T4", "AVG"],
        ["T5-AVG", "T5", "AVG"], ["P3-AVG", "P3", "AVG"], ["Pz-AVG", "Pz", "AVG"], ["P4-AVG", "P4", "AVG"],
        ["T6-AVG", "T6", "AVG"], ["O1-AVG", "O1", "AVG"], ["O2-AVG", "O2", "AVG"]
    ],
    "Linked ears reference": [
        ["Fp1-A1/A2", "Fp1", "A1A2"], ["Fp2-A1/A2", "Fp2", "A1A2"], ["F7-A1/A2", "F7", "A1A2"], ["F3-A1/A2", "F3", "A1A2"],
        ["Fz-A1/A2", "Fz", "A1A2"], ["F4-A1/A2", "F4", "A1A2"], ["F8-A1/A2", "F8", "A1A2"], ["T3-A1/A2", "T3", "A1A2"],
        ["C3-A1/A2", "C3", "A1A2"], ["Cz-A1/A2", "Cz", "A1A2"], ["C4-A1/A2", "C4", "A1A2"], ["T4-A1/A2", "T4", "A1A2"],
        ["T5-A1/A2", "T5", "A1A2"], ["P3-A1/A2", "P3", "A1A2"], ["Pz-A1/A2", "Pz", "A1A2"], ["P4-A1/A2", "P4", "A1A2"],
        ["T6-A1/A2", "T6", "A1A2"], ["O1-A1/A2", "O1", "A1A2"], ["O2-A1/A2", "O2", "A1A2"]
    ],
    "Raw Electrodes": [
        ["Fp1", "Fp1", "NONE"], ["Fp2", "Fp2", "NONE"], ["F7", "F7", "NONE"], ["F3", "F3", "NONE"],
        ["Fz", "Fz", "NONE"], ["F4", "F4", "NONE"], ["F8", "F8", "NONE"], ["T3", "T3", "NONE"],
        ["C3", "C3", "NONE"], ["Cz", "Cz", "NONE"], ["C4", "C4", "NONE"], ["T4", "T4", "NONE"],
        ["T5", "T5", "NONE"], ["P3", "P3", "NONE"], ["Pz", "Pz", "NONE"], ["P4", "P4", "NONE"],
        ["T6", "T6", "NONE"], ["O1", "O1", "NONE"], ["O2", "O2", "NONE"]
    ]
};
