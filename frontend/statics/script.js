document.getElementById("risk-form").addEventListener("submit", async function(e) {
    e.preventDefault();

    const formData = Object.fromEntries(new FormData(this).entries());
    
    
    const numericFields = ["age","height","weight","bp","glucose"];
    for (let key in formData) {
        if (numericFields.includes(key)) {
            formData[key] = Number(formData[key]);
        } // else keep strings for activity/diet/family_history
    }

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(formData)
        });
        const data = await response.json();

        const resultDiv = document.getElementById("result");
        if (data.error) {
            resultDiv.textContent = data.error;
            resultDiv.className = "";
        } else {
            const riskMap = {
                0: { text: "Low Risk (5 years)", class: "low" },
                1: { text: "Potrntial Risk (5 years)", class: "medium" },
                2: { text: "Medium Risk (5 years)", class: "medium" },
                3: { text: "High Risk (5 years)", class: "high" },
                4: { text: "Dangerous Risk (5 years)", class: "danger" }
            };
            const risk = riskMap[Number(data.risk_level)] || { text: "Unknown Risk Level", class: "unknown" };
            resultDiv.textContent = risk.text;
            resultDiv.className = risk.class;
        }
    } catch (err) {
        const resultDiv = document.getElementById("result");
        resultDiv.textContent = "Error contacting server.";
        resultDiv.className = "";
    }
});
