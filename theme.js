// manual light/dark toggle, overrides prefers-color-scheme; choice persists in localStorage
(function () {
  var root = document.documentElement;
  var KEY = "theme";

  function current() {
    return root.getAttribute("data-theme") ||
      (window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  }

  document.addEventListener("DOMContentLoaded", function () {
    var btn = document.getElementById("themeToggle");
    if (!btn) return;
    function sync() {
      var c = current();
      btn.textContent = c === "dark" ? "☾" : "☀";
      btn.setAttribute("aria-label", c === "dark" ? "Switch to light theme" : "Switch to dark theme");
    }
    btn.addEventListener("click", function () {
      var next = current() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem(KEY, next); } catch (e) {}
      sync();
    });
    sync();
  });
})();
