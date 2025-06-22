document.getElementById('form').addEventListener('submit', async function(e) {
    e.preventDefault();

    const formData = new FormData(e.target);
    const payload = {};
    formData.forEach((val, key) => payload[key] = isNaN(val) ? val : Number(val));

    const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const data = await response.json();
    document.getElementById("result").innerHTML = `
        <p><strong>Inorganic Fertilizer:</strong> ${data["Inorganic Recommendation"]}</p>
        <p><strong>Organic Alternative:</strong> ${data["Organic Alternative"]}</p>
    `;
});
