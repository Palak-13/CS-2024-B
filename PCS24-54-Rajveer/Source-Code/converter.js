function convertXMLToRSS() {
    const xmlFile = document.getElementById("xmlFile").files[0];

    if (xmlFile) {
        const reader = new FileReader();

        reader.onload = function (e) {
            const xmlData = e.target.result;
            const rssFeed = convertToRSS(xmlData);

            // Store the RSS feed in sessionStorage
            sessionStorage.setItem("rssFeed", rssFeed);

            // Redirect to the RSS feed display page
            window.location.href = "rss-display.html";
        };

        reader.readAsText(xmlFile);
    }
}

function convertToRSS(xmlData) {
    const parser = new DOMParser();
    const xmlDoc = parser.parseFromString(xmlData, "text/xml");

    if (xmlDoc.documentElement.nodeName === "parsererror") {
        return "Invalid XML";
    }

    // Create an RSS feed based on the XML structure
    let rssFeed = '<?xml version="1.0" encoding="UTF-8" ?><rss version="2.0"><channel>';

    const items = xmlDoc.querySelectorAll("*");

    items.forEach(item => {
        rssFeed += `<item>
            <title>${item.nodeName}</title>
            <description>${item.textContent}</description>
            <link>#</link>
        </item>`;
    });

    rssFeed += "</channel></rss>";

    return rssFeed;
}
