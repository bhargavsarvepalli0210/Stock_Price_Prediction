// script.js

document.addEventListener("DOMContentLoaded", function() {
    // Function to fetch data from the Flask backend
    function fetchData() {
        fetch('/data') // Flask route to get data
            .then(response => response.json()) // Parse the JSON response
            .then(data => {
                // Display data on the webpage
                document.getElementById('data-container').innerHTML = `
                    <p>Data from server:</p>
                    <pre>${JSON.stringify(data, null, 2)}</pre>
                `;
            })
            .catch(error => {
                console.error('Error fetching data:', error);
            });
    }

    // Check if the button element exists before adding the event listener
    var fetchButton = document.getElementById('fetch-button');
    if (fetchButton) {
        fetchButton.addEventListener('click', fetchData);
    } else {
        console.error('Fetch button not found.');
    }
});
