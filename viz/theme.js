/* Colour schemes for the viz pages (index.html and e2e.html).
 *
 * Every scheme is a full set of custom properties applied to <html>, and both
 * pages read only those properties — no component has a per-theme branch, and
 * the canvas painters resolve them once into a palette object (viz/e2e.js).
 * The default is Nord; the :root block in style.css carries the same values as
 * a fallback so a blocked script still renders instead of flashing unstyled.
 *
 * Loaded synchronously from <head> so the stored scheme is applied before the
 * first paint. The magnitude ramp hue/lightness are unitless numbers because
 * app.js builds `hsl()` from them.
 */
(function () {
  "use strict";

  const KEY = "minfer.viz.scheme";
  const DEFAULT = "nord";

  const SCHEMES = {
    // Warm bench instrument.
    graphite: {
      label: "Graphite",
      vars: {
        "--bg": "#121213",
        "--surface": "#171719",
        "--surface-2": "#1e1e22",
        "--raised": "#242427",
        "--raised-2": "#2b2b2f",
        "--sunken": "#0d0d0e",
        "--line": "rgba(236, 236, 239, 0.08)",
        "--line-2": "rgba(236, 236, 239, 0.14)",
        "--fg": "#ececed",
        "--fg-2": "#a6a6ad",
        "--fg-3": "#86868d",
        "--accent": "#e0a45c",
        "--accent-bright": "#f3cd97",
        "--accent-soft": "rgba(224, 164, 92, 0.14)",
        "--on-accent": "#241705",
        "--on-lit": "#241705",
        "--lit-edge": "rgba(36, 23, 5, 0.5)",
        "--metal": "#7d9cbe",
        "--cpu": "#bd7f5c",
        "--cuda": "#74b08d",
        "--none": "#6c6c74",
        "--ok": "#79b98f",
        "--bad": "#d97a6c",
        "--node-fill": "#1e1e22",
        "--grid": "rgba(236, 236, 239, 0.05)",
        "--edge": "rgba(236, 236, 239, 0.07)",
        "--edge-kv": "rgba(236, 236, 239, 0.12)",
        "--edge-residual": "rgba(236, 236, 239, 0.04)",
        "--ring": "#ececed",
        "--shadow": "rgba(0, 0, 0, 0.45)",
        "--ramp-h": "36",
        "--ramp-s": "62",
        "--ramp-lo": "52",
        "--ramp-hi": "86",
        "--heat-s": "50",
        "--heat-lo": "18",
        "--heat-hi": "80",
        "--chip-lit-bg": "rgba(36, 23, 5, 0.82)",
        "--chip-lit-line": "rgba(36, 23, 5, 0.9)",
        "--chip-lit-label": "#f3cd97",
        "--chip-pending": "rgba(36, 23, 5, 0.42)",
        "--chip-pending-text": "rgba(36, 23, 5, 0.72)",
        // shape-flow series (viz/e2e.js)
        "--series-act": "#e0a45c",
        "--series-logits": "#7d9cbe",
        "--series-probs": "#b48ead",
        "--series-kv": "#74b08d",
        "--series-ids": "#c9c0ad",
      },
    },

    // Nord (nordtheme.com), the default: polar-night chassis, frost accent.
    nord: {
      label: "Nord",
      vars: {
        "--bg": "#2e3440",
        "--surface": "#333b48",
        "--surface-2": "#3b4252",
        "--raised": "#434c5e",
        "--raised-2": "#4c566a",
        "--sunken": "#282e39",
        "--line": "rgba(216, 222, 233, 0.10)",
        "--line-2": "rgba(216, 222, 233, 0.18)",
        "--fg": "#eceff4",
        "--fg-2": "#d8dee9",
        "--fg-3": "#a0aabc",
        "--accent": "#88c0d0",
        "--accent-bright": "#a9d4e0",
        "--accent-soft": "rgba(136, 192, 208, 0.16)",
        "--on-accent": "#232a35",
        "--on-lit": "#1f2530",
        "--lit-edge": "rgba(31, 37, 48, 0.5)",
        "--metal": "#81a1c1",
        "--cpu": "#d08770",
        "--cuda": "#a3be8c",
        "--none": "#4c566a",
        "--ok": "#a3be8c",
        "--bad": "#bf616a",
        "--node-fill": "#3b4252",
        "--grid": "rgba(216, 222, 233, 0.05)",
        "--edge": "rgba(216, 222, 233, 0.08)",
        "--edge-kv": "rgba(216, 222, 233, 0.14)",
        "--edge-residual": "rgba(216, 222, 233, 0.05)",
        "--ring": "#eceff4",
        "--shadow": "rgba(0, 0, 0, 0.4)",
        "--ramp-h": "193",
        "--ramp-s": "48",
        "--ramp-lo": "46",
        "--ramp-hi": "82",
        "--heat-s": "45",
        "--heat-lo": "20",
        "--heat-hi": "82",
        "--chip-lit-bg": "rgba(31, 37, 48, 0.85)",
        "--chip-lit-line": "rgba(31, 37, 48, 0.9)",
        "--chip-lit-label": "#a9d4e0",
        "--chip-pending": "rgba(31, 37, 48, 0.42)",
        "--chip-pending-text": "rgba(31, 37, 48, 0.72)",
        "--series-act": "#88c0d0",
        "--series-logits": "#81a1c1",
        "--series-probs": "#b48ead",
        "--series-kv": "#a3be8c",
        "--series-ids": "#ebcb8b",
      },
    },

    // Dracula (draculatheme.com).
    dracula: {
      label: "Dracula",
      vars: {
        "--bg": "#282a36",
        "--surface": "#21222c",
        "--surface-2": "#2b2d3a",
        "--raised": "#343746",
        "--raised-2": "#44475a",
        "--sunken": "#1c1d25",
        "--line": "rgba(248, 248, 242, 0.10)",
        "--line-2": "rgba(248, 248, 242, 0.18)",
        "--fg": "#f8f8f2",
        "--fg-2": "#cdd0dd",
        "--fg-3": "#a1a4bd",
        "--accent": "#bd93f9",
        "--accent-bright": "#d6b8ff",
        "--accent-soft": "rgba(189, 147, 249, 0.16)",
        "--on-accent": "#1e1b2e",
        "--on-lit": "#1b1830",
        "--lit-edge": "rgba(27, 24, 48, 0.5)",
        "--metal": "#8be9fd",
        "--cpu": "#ffb86c",
        "--cuda": "#50fa7b",
        "--none": "#6272a4",
        "--ok": "#50fa7b",
        "--bad": "#ff5555",
        "--node-fill": "#2b2d3a",
        "--grid": "rgba(248, 248, 242, 0.05)",
        "--edge": "rgba(248, 248, 242, 0.09)",
        "--edge-kv": "rgba(248, 248, 242, 0.16)",
        "--edge-residual": "rgba(248, 248, 242, 0.05)",
        "--ring": "#f8f8f2",
        "--shadow": "rgba(0, 0, 0, 0.5)",
        "--ramp-h": "265",
        "--ramp-s": "65",
        "--ramp-lo": "50",
        "--ramp-hi": "84",
        "--heat-s": "62",
        "--heat-lo": "20",
        "--heat-hi": "84",
        "--chip-lit-bg": "rgba(27, 24, 48, 0.85)",
        "--chip-lit-line": "rgba(27, 24, 48, 0.9)",
        "--chip-lit-label": "#d6b8ff",
        "--chip-pending": "rgba(27, 24, 48, 0.42)",
        "--chip-pending-text": "rgba(27, 24, 48, 0.72)",
        "--series-act": "#bd93f9",
        "--series-logits": "#8be9fd",
        "--series-probs": "#ff79c6",
        "--series-kv": "#50fa7b",
        "--series-ids": "#f1fa8c",
      },
    },

    // Cold monitoring desk.
    ink: {
      label: "Ink",
      vars: {
        "--bg": "#0b0d11",
        "--surface": "#101319",
        "--surface-2": "#171b23",
        "--raised": "#1d222c",
        "--raised-2": "#242b38",
        "--sunken": "#080a0e",
        "--line": "rgba(226, 234, 243, 0.08)",
        "--line-2": "rgba(226, 234, 243, 0.15)",
        "--fg": "#e6eaf0",
        "--fg-2": "#9fa9ba",
        "--fg-3": "#7a8496",
        "--accent": "#59c4e2",
        "--accent-bright": "#a6e6f6",
        "--accent-soft": "rgba(89, 196, 226, 0.14)",
        "--on-accent": "#04161d",
        "--on-lit": "#04161d",
        "--lit-edge": "rgba(4, 22, 29, 0.5)",
        "--metal": "#8aa6c6",
        "--cpu": "#c58a5c",
        "--cuda": "#78b894",
        "--none": "#6b7280",
        "--ok": "#6cc39a",
        "--bad": "#e0796f",
        "--node-fill": "#171b23",
        "--grid": "rgba(226, 234, 243, 0.05)",
        "--edge": "rgba(226, 234, 243, 0.07)",
        "--edge-kv": "rgba(226, 234, 243, 0.12)",
        "--edge-residual": "rgba(226, 234, 243, 0.04)",
        "--ring": "#e6eaf0",
        "--shadow": "rgba(0, 0, 0, 0.5)",
        "--ramp-h": "190",
        "--ramp-s": "60",
        "--ramp-lo": "42",
        "--ramp-hi": "78",
        "--heat-s": "55",
        "--heat-lo": "16",
        "--heat-hi": "78",
        "--chip-lit-bg": "rgba(4, 22, 29, 0.85)",
        "--chip-lit-line": "rgba(4, 22, 29, 0.9)",
        "--chip-lit-label": "#a6e6f6",
        "--chip-pending": "rgba(4, 22, 29, 0.42)",
        "--chip-pending-text": "rgba(4, 22, 29, 0.72)",
        "--series-act": "#59c4e2",
        "--series-logits": "#8aa6c6",
        "--series-probs": "#c58a5c",
        "--series-kv": "#78b894",
        "--series-ids": "#d9c58a",
      },
    },

    // Light: warm paper, copper ink. The one scheme where "hot" is darker.
    paper: {
      label: "Paper",
      vars: {
        "--bg": "#f3efe6",
        "--surface": "#fbf9f4",
        "--surface-2": "#f0ebe1",
        "--raised": "#e9e3d6",
        "--raised-2": "#e0d9c9",
        "--sunken": "#eae4d8",
        "--line": "rgba(31, 27, 22, 0.1)",
        "--line-2": "rgba(31, 27, 22, 0.18)",
        "--fg": "#1f1b16",
        "--fg-2": "#5b5448",
        "--fg-3": "#7b7266",
        "--accent": "#a85a28",
        "--accent-bright": "#c9783d",
        "--accent-soft": "rgba(168, 90, 40, 0.16)",
        "--on-accent": "#fdf8f0",
        "--on-lit": "#241305",
        "--lit-edge": "rgba(36, 19, 5, 0.45)",
        "--metal": "#47698c",
        "--cpu": "#9c542b",
        "--cuda": "#3f7a5b",
        "--none": "#7b7266",
        "--ok": "#2f7d4f",
        "--bad": "#b4382c",
        "--node-fill": "#fdfbf7",
        "--grid": "rgba(31, 27, 22, 0.07)",
        "--edge": "rgba(31, 27, 22, 0.13)",
        "--edge-kv": "rgba(31, 27, 22, 0.18)",
        "--edge-residual": "rgba(31, 27, 22, 0.05)",
        "--ring": "#1f1b16",
        "--shadow": "rgba(31, 27, 22, 0.16)",
        "--ramp-h": "32",
        "--ramp-s": "60",
        "--ramp-lo": "64",
        "--ramp-hi": "90",
        "--heat-s": "45",
        "--heat-lo": "92",
        "--heat-hi": "30",
        "--chip-lit-bg": "rgba(31, 18, 6, 0.85)",
        "--chip-lit-line": "rgba(31, 18, 6, 0.9)",
        "--chip-lit-label": "#f6e8d4",
        "--chip-pending": "rgba(31, 18, 6, 0.4)",
        "--chip-pending-text": "rgba(31, 18, 6, 0.7)",
        "--series-act": "#a85a28",
        "--series-logits": "#47698c",
        "--series-probs": "#8a4f74",
        "--series-kv": "#3f7a5b",
        "--series-ids": "#8a6a1f",
      },
    },

    // Monochrome. The only hue left on the page is the backend's top rule.
    noir: {
      label: "Noir",
      vars: {
        "--bg": "#0a0a0b",
        "--surface": "#101012",
        "--surface-2": "#171719",
        "--raised": "#1e1e21",
        "--raised-2": "#26262a",
        "--sunken": "#070708",
        "--line": "rgba(240, 240, 242, 0.08)",
        "--line-2": "rgba(240, 240, 242, 0.15)",
        "--fg": "#f0f0f2",
        "--fg-2": "#a5a5ad",
        "--fg-3": "#808088",
        "--accent": "#d6d6da",
        "--accent-bright": "#ffffff",
        "--accent-soft": "rgba(214, 214, 218, 0.12)",
        "--on-accent": "#0a0a0b",
        "--on-lit": "#0a0a0b",
        "--lit-edge": "rgba(10, 10, 11, 0.45)",
        "--metal": "#8fa6c0",
        "--cpu": "#c08a5e",
        "--cuda": "#7fb894",
        "--none": "#6c6c74",
        "--ok": "#86c3a0",
        "--bad": "#e07a72",
        "--node-fill": "#171719",
        "--grid": "rgba(240, 240, 242, 0.045)",
        "--edge": "rgba(240, 240, 242, 0.07)",
        "--edge-kv": "rgba(240, 240, 242, 0.12)",
        "--edge-residual": "rgba(240, 240, 242, 0.04)",
        "--ring": "#ffffff",
        "--shadow": "rgba(0, 0, 0, 0.55)",
        "--ramp-h": "0",
        "--ramp-s": "0",
        "--ramp-lo": "52",
        "--ramp-hi": "92",
        "--heat-s": "0",
        "--heat-lo": "12",
        "--heat-hi": "92",
        "--chip-lit-bg": "rgba(10, 10, 11, 0.85)",
        "--chip-lit-line": "rgba(10, 10, 11, 0.9)",
        "--chip-lit-label": "#f0f0f2",
        "--chip-pending": "rgba(10, 10, 11, 0.42)",
        "--chip-pending-text": "rgba(10, 10, 11, 0.72)",
        "--series-act": "#f0f0f2",
        "--series-logits": "#c8c8cc",
        "--series-probs": "#a0a0a6",
        "--series-kv": "#82828a",
        "--series-ids": "#e4e4e6",
      },
    },
  };

  const ORDER = ["nord", "graphite", "dracula", "ink", "paper", "noir"];
  let current = DEFAULT;
  const applied = new Set();

  function readStored() {
    try {
      return localStorage.getItem(KEY) || DEFAULT;
    } catch (e) {
      return DEFAULT; // file:// or blocked storage: the default is fine
    }
  }

  function apply(id, persist) {
    const name = SCHEMES[id] ? id : DEFAULT;
    const root = document.documentElement;
    for (const k of applied) root.style.removeProperty(k);
    applied.clear();
    const vars = SCHEMES[name].vars;
    for (const k in vars) {
      root.style.setProperty(k, vars[k]);
      applied.add(k);
    }
    root.dataset.scheme = name;
    current = name;
    if (persist) {
      try { localStorage.setItem(KEY, name); } catch (e) { /* keep the session */ }
    }
    document.dispatchEvent(new CustomEvent("minfer-theme", { detail: { scheme: name } }));
  }

  function mount() {
    const selects = document.querySelectorAll("select[data-scheme]");
    selects.forEach(sel => {
      sel.textContent = "";
      for (const id of ORDER) {
        const opt = document.createElement("option");
        opt.value = id;
        opt.textContent = SCHEMES[id].label;
        sel.appendChild(opt);
      }
      sel.value = current;
      sel.addEventListener("change", () => apply(sel.value, true));
    });
  }

  apply(readStored(), false);

  window.minferTheme = {
    schemes: () => ORDER.map(id => ({ id, label: SCHEMES[id].label })),
    get: () => current,
    set: id => apply(id, true),
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
  else mount();
})();
