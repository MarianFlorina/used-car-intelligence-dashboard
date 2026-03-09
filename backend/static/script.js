let priceChart = null;



const currencySelect = document.getElementById("currencySelect");
const priceHint = document.getElementById("priceHint");

const currencyExamples = {
    USD: "$15000",
    INR: "₹1200000",
    EUR: "€14000",
    GBP: "£12000"
};

currencySelect.addEventListener("change", () => {

    const currency = currencySelect.value;
    priceHint.innerText = "Example: " + currencyExamples[currency];

});




document.getElementById("predictForm").addEventListener("submit", function (e) {

    e.preventDefault();



    const yearInput = document.querySelector("select[name='year']");
    const mileageInput = document.querySelector("input[name='odometer']");
    const manufacturerInput = document.querySelector("input[name='manufacturer']");
    const listedPriceInput = document.querySelector("input[name='listed_price']");

    const fuelInput = document.querySelector("select[name='fuel']");
    const transmissionInput = document.querySelector("select[name='transmission']");
    const driveInput = document.querySelector("select[name='drive']");
    const typeInput = document.querySelector("select[name='type']");
    const conditionInput = document.querySelector("select[name='condition']");
    const currencyInput = document.querySelector("select[name='currency']");

    const button = document.querySelector("button");




    const resultBox = document.getElementById("resultBox");
    const errorBox = document.getElementById("errorBox");

    resultBox.classList.add("hidden");
    errorBox.classList.add("hidden");




    const year = parseInt(yearInput.value);
    const mileage = parseInt(mileageInput.value);
    const manufacturer = manufacturerInput.value.trim();
    const listed_price = listedPriceInput.value;

    const currentYear = new Date().getFullYear();




    if (isNaN(year) || year < 1980 || year > currentYear) {
        showError("Manufacturing year must be between 1980 and " + currentYear);
        return;
    }

    if (isNaN(mileage) || mileage < 0 || mileage > 1000000) {
        showError("Mileage must be between 0 and 1,000,000 miles");
        return;
    }

    if (manufacturer.length < 2 || !/^[a-zA-Z]+$/.test(manufacturer)) {
        showError("Manufacturer must contain only alphabets");
        return;
    }




    const payload = {

        year: year,
        odometer: mileage,
        manufacturer: manufacturer,
        listed_price: listed_price,
        fuel: fuelInput.value,
        transmission: transmissionInput.value,
        drive: driveInput.value,
        type: typeInput.value,
        condition: conditionInput.value,
        currency: currencyInput.value

    };




    button.innerText = "Analyzing Vehicle...";
    button.disabled = true;




    fetch("/predict", {

        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)

    })

    .then(res => res.json())

    .then(data => {

        if (data.error) {
            showError(data.error);
            return;
        }




        document.getElementById("predictedPrice").innerText =
            `${data.currency} ${data.predicted_price}`;

        document.getElementById("vehicleAge").innerText =
            `${data.vehicle_age} years`;

        document.getElementById("dealStatus").innerText =
            data.deal_status || "N/A";

        document.getElementById("recommendation").innerText =
            data.recommendation || "N/A";




        document.getElementById("confidenceScore").innerText =
            data.confidence + "%";

        document.getElementById("priceRange").innerText =
            `${data.currency} ${data.price_range[0]} – ${data.price_range[1]}`;

        document.getElementById("dashDealStatus").innerText =
            data.deal_status || "N/A";

        document.getElementById("dashRecommendation").innerText =
            data.recommendation || "N/A";




        const ctx = document.getElementById("priceChart");

        if (priceChart) {
            priceChart.destroy();
        }

        priceChart = new Chart(ctx, {

            type: "bar",

            data: {
                labels: ["Predicted Price", "Lower Range", "Upper Range"],

                datasets: [{
                    data: [
                        data.predicted_price,
                        data.price_range[0],
                        data.price_range[1]
                    ],

                    backgroundColor: [
                        "#667eea",
                        "#2ecc71",
                        "#f39c12"
                    ]
                }]
            },

            options: {
                responsive: true,
                plugins: {
                    legend: { display: false }
                }
            }

        });




        const table = document.getElementById("similarCarsTable");
        table.innerHTML = "";

        if (data.similar_cars.length === 0) {

            table.innerHTML =
                "<tr><td colspan='4'>No similar cars found</td></tr>";

        } else {

            data.similar_cars.forEach(car => {

                const row = document.createElement("tr");

                row.innerHTML = `
                    <td>${car.manufacturer}</td>
                    <td>${car.year}</td>
                    <td>${car.odometer}</td>
                    <td>${car.price}</td>
                `;

                table.appendChild(row);

            });

        }




        resultBox.classList.remove("hidden");

    })


    .catch(() => {

        showError("Prediction failed. Please check inputs.");

    })


    .finally(() => {

        button.innerText = "Analyze Vehicle";
        button.disabled = false;

    });

});




function showError(message) {

    const errorBox = document.getElementById("errorBox");

    errorBox.innerText = message;

    errorBox.classList.remove("hidden");

}