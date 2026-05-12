/* OpenClaw Dashboard — HTMX config + confirm dialogs */

document.addEventListener("DOMContentLoaded", function () {
    /* HTMX: add confirm dialog for destructive actions */
    document.body.addEventListener("htmx:confirm", function (e) {
        var msg = e.detail.elt.getAttribute("hx-confirm");
        if (msg) {
            e.preventDefault();
            if (confirm(msg)) {
                e.detail.issueRequest();
            }
        }
    });

    /* Auto-refresh scores every 60s on watchlist page */
    var scoreTable = document.getElementById("score-table");
    if (scoreTable) {
        setInterval(function () {
            htmx.trigger(scoreTable, "refresh");
        }, 60000);
    }
});
