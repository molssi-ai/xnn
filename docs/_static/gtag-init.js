/* Google Analytics 4 (loaded via html_js_files in conf.py).
 *
 * Injected directly rather than through pydata-sphinx-theme's
 * html_theme_options["analytics"], which hardcodes
 *   gtag('consent', 'default', {... 'analytics_storage': 'denied' ...})
 * with no way to override it. Under that default GA4 sends cookieless pings
 * only and the standard reports stay empty. Setting both would also inject
 * the snippet twice.
 */
window.dataLayer = window.dataLayer || [];
function gtag() { dataLayer.push(arguments); }
gtag("js", new Date());
gtag("config", "G-YK7FCTPX7M");
