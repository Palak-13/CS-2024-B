// $(document).ready(function () {
//     $("#convertButton").click(function () {
//         var fileInput = document.getElementById("xmlFileInput");
//         var file = fileInput.files[0];

//         if (file) {
//             var reader = new FileReader();

//             reader.onload = function (e) {
//                 var xmlData = e.target.result;
//                 var rssData = convertXMLtoRSS(xmlData); // Make sure this function generates valid RSS content
//                 openNewWindowWithRSS(rssData);
//             };

//             reader.readAsText(file);
//         } else {
//             alert("Please select an XML file.");
//         }
//     });

//     function convertXMLtoRSS(xmlData) {
//         // Your XML to RSS conversion logic here (replace with your code)
//         // Make sure the conversion is done correctly and the result is a valid RSS feed content
//         // For example, the content should start with "<?xml version="1.0" encoding="UTF-8" ?>"

//         var rssData = `
//             <?xml version="1.0" encoding="UTF-8" ?>
//             <rss version="2.0">
//                 <channel>
//                     <title>My RSS Feed</title>
//                     <link>https://example.com</link>
//                     <description>Converted RSS feed from XML</description>
//                     <item>
//                         <title>Sample RSS Item</title>
//                         <link>https://example.com/sample-item</link>
//                         <description>This is a sample item in the RSS feed.</description>
//                     </item>
//                 </channel>
//             </rss>
//         `;

//         return rssData;
//     }

//     function openNewWindowWithRSS(rssData) {
//         var rssWindow = window.open("", "_blank");
//         rssWindow.document.write(`<html><head><link rel="stylesheet" type="text/css" href="rss-style.css"></head><body>${rssData}</body></html>`);
//         rssWindow.document.close();
//     }
// });
