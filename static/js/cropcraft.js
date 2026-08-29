// Theme toggle, persisted across visits and defaulting to the OS preference.
// Replaces the jQuery 1.11 + templatemo script pair from CropCraft 1.x.
(() => {
  "use strict";

  const root = document.documentElement;
  const STORAGE_KEY = "cropcraft-theme";

  const preferred = () =>
    localStorage.getItem(STORAGE_KEY) ??
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

  const apply = (theme) => {
    root.setAttribute("data-bs-theme", theme);
    const button = document.getElementById("cc-theme-toggle");
    if (button) {
      button.firstElementChild.innerHTML = theme === "dark" ? "&#9788;" : "&#9789;";
    }
  };

  apply(preferred());

  document.addEventListener("DOMContentLoaded", () => {
    apply(preferred());
    document.getElementById("cc-theme-toggle")?.addEventListener("click", () => {
      const next = root.getAttribute("data-bs-theme") === "dark" ? "light" : "dark";
      localStorage.setItem(STORAGE_KEY, next);
      apply(next);
    });
  });
})();

// Pre-select the crop when arriving from a crop recommendation.
document.addEventListener("DOMContentLoaded", () => {
  const select = document.getElementById("cropname");
  const requested = new URLSearchParams(window.location.search).get("crop");
  if (select && requested && [...select.options].some((o) => o.value === requested)) {
    select.value = requested;
  }
});

// Preview the chosen pest photo locally. Nothing is uploaded until submit.
document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("image");
  const preview = document.getElementById("cc-preview");
  if (!input || !preview) return;

  input.addEventListener("change", (event) => {
    const [file] = event.target.files;
    if (preview.src.startsWith("blob:")) URL.revokeObjectURL(preview.src);
    if (!file) {
      preview.classList.add("d-none");
      return;
    }
    preview.src = URL.createObjectURL(file);
    preview.classList.remove("d-none");
  });
});

// Hide images that fail to load, replacing inline onerror handlers so the page
// can run under a Content-Security-Policy that forbids inline script.
document.addEventListener("DOMContentLoaded", () => {
  for (const image of document.querySelectorAll("img[data-hide-on-error]")) {
    image.addEventListener("error", () => image.classList.add("d-none"), { once: true });
  }
});
