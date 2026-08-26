let priceChart = null;
let modelChart = null;
let shapChart = null;
let depreciationChart = null;
let trendChart = null;
let regionalChart = null;
let savedCars = [];
let liveData = null;

// Fetch live data on page load
fetchLiveData();

function fetchLiveData() {
    fetch("/live_data")
    .then(res => res.json())
    .then(data => {
        liveData = data;
        const banner = document.getElementById("liveDataBanner");
        const status = document.getElementById("liveDataStatus");
        const source = document.getElementById("liveDataSource");

        status.textContent = `Live rates: $1 = ₹${data.currency_rates.INR?.toFixed(2)} | €${data.currency_rates.EUR?.toFixed(4)} | £${data.currency_rates.GBP?.toFixed(4)}`;
        source.textContent = `Fuel: ${data.fuel_prices.source} (${data.fuel_prices.gasoline.toFixed(2)}/gal)`;

        banner.style.background = "#e8f8f0";
    })
    .catch(() => {
        const status = document.getElementById("liveDataStatus");
        status.textContent = "Using cached rates";
    });
}

const currencySelect = document.getElementById("currencySelect");
const priceHint = document.getElementById("priceHint");
const currencyExamples = { USD: "$15000", INR: "1200000", EUR: "14000", GBP: "12000" };

currencySelect.addEventListener("change", () => {
    priceHint.innerText = "Example: " + currencyExamples[currencySelect.value];
});

document.getElementById("predictForm").addEventListener("submit", function (e) {
    e.preventDefault();

    const yearInput = document.querySelector("select[name='year']");
    const mileageInput = document.querySelector("input[name='odometer']");
    const manufacturerInput = document.querySelector("input[name='manufacturer']");
    const listedPriceInput = document.querySelector("input[name='listed_price']");
    const button = document.querySelector("button[type='submit']");
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
        year: year, odometer: mileage, manufacturer: manufacturer,
        listed_price: listed_price,
        fuel: document.querySelector("select[name='fuel']").value,
        transmission: document.querySelector("select[name='transmission']").value,
        drive: document.querySelector("select[name='drive']").value,
        type: document.querySelector("select[name='type']").value,
        condition: document.querySelector("select[name='condition']").value,
        currency: currencySelect.value
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
        if (data.error) { showError(data.error); return; }

        // Basic results
        document.getElementById("predictedPrice").innerText = data.symbol + " " + data.predicted_price.toLocaleString();
        document.getElementById("vehicleAge").innerText = data.vehicle_age + " years";
        document.getElementById("dealStatus").innerText = data.deal_status;
        document.getElementById("recommendation").innerText = data.recommendation;
        document.getElementById("confidenceScore").innerText = data.confidence + "%";
        document.getElementById("priceRange").innerText = data.symbol + " " + data.price_range[0].toLocaleString() + " - " + data.price_range[1].toLocaleString();
        document.getElementById("dashDealStatus").innerText = data.deal_status;
        document.getElementById("dashRecommendation").innerText = data.recommendation;
        resultBox.classList.remove("hidden");

        // Model Comparison Chart
        updateModelChart(data.model_comparison, data.symbol);

        // SHAP Waterfall
        updateShapChart(data.shap_values);

        // Depreciation Curve
        updateDepreciationChart(data.depreciation, data.symbol, data.vehicle_age);

        // TCO
        if (data.tco) {
            document.getElementById("tcoFuel").innerText = data.symbol + " " + data.tco.annual_fuel.toLocaleString();
            document.getElementById("tcoInsurance").innerText = data.symbol + " " + data.tco.annual_insurance.toLocaleString();
            document.getElementById("tcoMaintenance").innerText = data.symbol + " " + data.tco.annual_maintenance.toLocaleString();
            document.getElementById("tcoMonthly").innerText = data.symbol + " " + data.tco.monthly_cost.toLocaleString();
            document.getElementById("tcoTotal").innerText = data.symbol + " " + data.tco.total_5yr.toLocaleString();
        }

        // Negotiation
        if (data.negotiation) {
            document.getElementById("negotiationBox").classList.remove("hidden");
            document.getElementById("negListed").innerText = data.symbol + " " + data.negotiation.listed_price.toLocaleString();
            document.getElementById("negFair").innerText = data.symbol + " " + data.negotiation.fair_price.toLocaleString();
            document.getElementById("negOffer").innerText = data.symbol + " " + data.negotiation.suggested_offer.toLocaleString();
            document.getElementById("negWalk").innerText = data.symbol + " " + data.negotiation.walk_away_price.toLocaleString();
            const tipsList = document.getElementById("negTips");
            tipsList.innerHTML = "";
            data.negotiation.strategy.forEach(tip => {
                const li = document.createElement("li");
                li.textContent = tip;
                tipsList.appendChild(li);
            });
        } else {
            document.getElementById("negotiationBox").classList.add("hidden");
        }

        // Price Comparison
        updatePriceChart(data.predicted_price, data.price_range, data.symbol);

        // Market Intelligence
        if (data.market) {
            document.getElementById("marketAvg").innerText = data.symbol + " " + data.market.avg_price.toLocaleString();
            document.getElementById("marketMedian").innerText = data.symbol + " " + data.market.median_price.toLocaleString();
            document.getElementById("marketListings").innerText = data.market.total_listings;
            document.getElementById("marketMileage").innerText = Math.round(data.market.avg_mileage).toLocaleString() + " mi";
            updateTrendChart(data.market.price_trend, data.symbol);
        }

        // Regional Comparison
        if (data.regional) {
            updateRegionalChart(data.regional, data.symbol);
        }

        // Similar Cars
        const table = document.getElementById("similarCarsTable");
        table.innerHTML = "";
        if (data.similar_cars.length === 0) {
            table.innerHTML = "<tr><td colspan='4'>No similar cars found</td></tr>";
        } else {
            data.similar_cars.forEach(car => {
                const row = document.createElement("tr");
                row.innerHTML = `<td>${car.manufacturer}</td><td>${car.year}</td><td>${car.odometer.toLocaleString()}</td><td>${data.symbol} ${car.price.toLocaleString()}</td>`;
                table.appendChild(row);
            });
        }

        // Save to compare
        saveCar(data, payload);
    })
    .catch(() => showError("Prediction failed. Please check inputs."))
    .finally(() => { button.innerText = "Analyze Vehicle"; button.disabled = false; });
});

function showError(message) {
    document.getElementById("errorBox").innerText = message;
    document.getElementById("errorBox").classList.remove("hidden");
}

function updateModelChart(comparison, symbol) {
    const ctx = document.getElementById("modelChart");
    if (modelChart) modelChart.destroy();
    modelChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Linear Regression", "Random Forest", "XGBoost", "Ensemble"],
            datasets: [{
                data: [comparison.linear, comparison.random_forest, comparison.xgboost, comparison.ensemble],
                backgroundColor: ["#95a5a6", "#3498db", "#e74c3c", "#2ecc71"]
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false }, title: { display: true, text: "Predicted Price by Model (" + symbol + ")" } },
            scales: { y: { beginAtZero: true } }
        }
    });
}

function updateShapChart(shapValues) {
    const ctx = document.getElementById("shapChart");
    if (shapChart) shapChart.destroy();
    if (!shapValues || shapValues.length === 0) return;

    const labels = shapValues.map(v => v.feature);
    const values = shapValues.map(v => v.value);
    const colors = shapValues.map(v => v.direction === "up" ? "#e74c3c" : "#2ecc71");

    shapChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            plugins: { legend: { display: false }, title: { display: true, text: "Feature Impact on Price (SHAP)" } },
            scales: { x: { title: { display: true, text: "Impact on log(price)" } } }
        }
    });
}

function updateDepreciationChart(points, symbol, currentAge) {
    const ctx = document.getElementById("depreciationChart");
    if (depreciationChart) depreciationChart.destroy();
    if (!points || points.length === 0) return;

    const labels = points.map(p => p.year.toString());
    const prices = points.map(p => p.predicted_price);
    const bgColors = points.map(p => p.age === currentAge ? "#e74c3c" : "#667eea");

    depreciationChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: "Predicted Value",
                data: prices,
                borderColor: "#667eea",
                backgroundColor: "rgba(102,126,234,0.1)",
                fill: true,
                pointBackgroundColor: bgColors,
                pointRadius: points.map(p => p.age === currentAge ? 8 : 3),
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: { display: true, text: "Depreciation Over Time" },
                legend: { display: false }
            },
            scales: {
                y: { title: { display: true, text: symbol + " Price" }, beginAtZero: true },
                x: { title: { display: true, text: "Model Year" } }
            }
        }
    });
}

function updatePriceChart(predicted, range, symbol) {
    const ctx = document.getElementById("priceChart");
    if (priceChart) priceChart.destroy();
    priceChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Predicted Price", "Lower Range", "Upper Range"],
            datasets: [{
                data: [predicted, range[0], range[1]],
                backgroundColor: ["#667eea", "#2ecc71", "#f39c12"]
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false }, title: { display: true, text: "Price Range (" + symbol + ")" } },
            scales: { y: { beginAtZero: true } }
        }
    });
}

function updateTrendChart(trend, symbol) {
    const ctx = document.getElementById("trendChart");
    if (trendChart) trendChart.destroy();
    if (!trend || Object.keys(trend).length === 0) return;

    const sorted = Object.entries(trend).sort((a, b) => a[0] - b[0]);
    const labels = sorted.map(([year]) => year);
    const prices = sorted.map(([, price]) => price);

    trendChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: "Avg Price",
                data: prices,
                borderColor: "#e74c3c",
                backgroundColor: "rgba(231,76,60,0.1)",
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            plugins: { title: { display: true, text: "Average Price by Model Year" } },
            scales: {
                y: { title: { display: true, text: symbol + " Price" }, beginAtZero: true },
                x: { title: { display: true, text: "Model Year" } }
            }
        }
    });
}

function updateRegionalChart(regional, symbol) {
    const ctx = document.getElementById("regionalChart");
    if (regionalChart) regionalChart.destroy();
    if (!regional || Object.keys(regional).length === 0) return;

    const labels = Object.keys(regional).map(k => k.charAt(0).toUpperCase() + k.slice(1));
    const prices = Object.values(regional).map(v => v.avg_price);

    regionalChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Avg Price",
                data: prices,
                backgroundColor: ["#2ecc71", "#3498db", "#f39c12"]
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false }, title: { display: true, text: "Average Price by Condition" } },
            scales: { y: { beginAtZero: true } }
        }
    });
}

// Save & Compare
function saveCar(data, inputs) {
    const car = {
        id: Date.now(),
        manufacturer: inputs.manufacturer,
        year: inputs.year,
        odometer: inputs.odometer,
        predicted: data.predicted_price,
        symbol: data.symbol,
        confidence: data.confidence,
        deal: data.deal_status
    };
    savedCars.push(car);
    renderCompareList();
}

function renderCompareList() {
    const list = document.getElementById("compareList");
    list.innerHTML = "";
    savedCars.forEach(car => {
        const card = document.createElement("div");
        card.className = "compare-card";
        card.innerHTML = `
            <button class="remove-btn" onclick="removeCar(${car.id})">&times;</button>
            <strong>${car.manufacturer} ${car.year}</strong><br>
            <small>${car.odometer.toLocaleString()} mi</small><br>
            <strong>${car.symbol} ${car.predicted.toLocaleString()}</strong><br>
            <small>Confidence: ${car.confidence}%</small><br>
            <small>${car.deal}</small>
        `;
        list.appendChild(card);
    });
}

function removeCar(id) {
    savedCars = savedCars.filter(c => c.id !== id);
    renderCompareList();
}