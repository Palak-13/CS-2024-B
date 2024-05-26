document.addEventListener("DOMContentLoaded", function () {
    // Retrieve the RSS feed from sessionStorage
    const rssFeed = sessionStorage.getItem("rssFeed");

    if (rssFeed) {
        // Display the RSS feed
        document.getElementById("rssFeedDisplay").innerHTML = rssFeed;
    }
});
