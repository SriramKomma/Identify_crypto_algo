const form = document.getElementById("predictForm");
const textArea = document.getElementById("ciphertext");
const resultCard = document.getElementById("resultCard");
const predictedAlgo = document.getElementById("predictedAlgo");
const errorBox = document.getElementById("errorBox");
let probChart = null;

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const ciphertext = textArea.value.trim();
  if (!ciphertext) {
    showError("Please enter ciphertext text.");
    return;
  }
  hideError();
  resultCard.classList.add("d-none");
  predictedAlgo.textContent = "Predicting...";
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ciphertext }),
    });
    const data = await response.json();
    if (response.ok) renderResults(data);
    else showError(data.error || "Server error");
  } catch (err) {
    showError(err.message);
  }
});

document.getElementById("clearBtn").addEventListener("click", () => {
  textArea.value = "";
  resultCard.classList.add("d-none");
  hideError();
});

function renderResults(data) {
  const hybrid = data.hybrid;
  const cnn = data.cnn;
  let algo = null;
  let probs = null;

  if (hybrid && hybrid.predicted) {
    algo = hybrid.predicted;
    probs = hybrid.results;
  } else if (cnn && cnn.predicted) {
    algo = cnn.predicted + " (CNN)";
    probs = cnn.scores.reduce((acc, v, i) => {
      acc["Class " + i] = (v * 100).toFixed(2);
      return acc;
    }, {});
  } else {
    showError("No prediction available.");
    return;
  }

  predictedAlgo.textContent = `Predicted: ${algo}`;
  resultCard.classList.remove("d-none");

  const labels = Object.keys(probs);
  const values = Object.values(probs);

  if (probChart) probChart.destroy();
  const ctx = document.getElementById("probChart").getContext("2d");
  probChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Probability (%)",
          data: values,
          borderWidth: 1,
        },
      ],
    },
    options: {
      scales: {
        y: { beginAtZero: true, max: 100 },
      },
    },
  });
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove("d-none");
}

function hideError() {
  errorBox.classList.add("d-none");
}
