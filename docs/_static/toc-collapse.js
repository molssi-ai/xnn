/* Collapse/expand the secondary (right-hand) sidebar.
 *
 * pydata-sphinx-theme has no built-in collapse for the table of contents --
 * only for the primary sidebar -- so this supplies the behaviour for
 * _templates/sidebar-secondary-collapse.html. The state is remembered per
 * browser so the choice survives navigation between pages.
 */
(function () {
  "use strict";

  var STORAGE_KEY = "xnn-toc-collapsed";
  var COLLAPSED_CLASS = "xnn-toc-squeeze";

  function apply(sidebar, button, collapsed) {
    sidebar.classList.toggle(COLLAPSED_CLASS, collapsed);
    button.setAttribute("aria-expanded", collapsed ? "false" : "true");
    button.setAttribute(
      "title",
      collapsed ? "Expand table of contents" : "Collapse table of contents"
    );
  }

  function init() {
    var sidebar = document.getElementById("pst-secondary-sidebar");
    var button = document.getElementById("xnn-collapse-toc-button");
    if (!sidebar || !button) {
      return;
    }

    // Reading storage can throw when site data is blocked; the sidebar simply
    // starts expanded in that case.
    var collapsed = false;
    try {
      collapsed = window.localStorage.getItem(STORAGE_KEY) === "true";
    } catch (e) {
      collapsed = false;
    }
    apply(sidebar, button, collapsed);

    button.addEventListener("click", function () {
      collapsed = !sidebar.classList.contains(COLLAPSED_CLASS);
      apply(sidebar, button, collapsed);
      try {
        window.localStorage.setItem(STORAGE_KEY, String(collapsed));
      } catch (e) {
        /* not persisting is not fatal */
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
