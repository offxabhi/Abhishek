document.addEventListener('DOMContentLoaded', function() {
    const predictBtn = document.getElementById('predictBtn');
    const stockSymbolInput = document.getElementById('stockSymbol');
    const loadingElement = document.getElementById('loading');
    const resultsElement = document.getElementById('results');
    const errorElement = document.getElementById('error');
    
    let stockChart = null;
    
    predictBtn.addEventListener('click', function() {
        const symbol = stockSymbolInput.value.trim();
        
        if (!symbol) {
            showError('Please enter a stock symbol');
            return;
        }
        
        // Show loading, hide results and error
        loadingElement.classList.remove('hidden');
        resultsElement.classList.add('hidden');
        errorElement.classList.add('hidden');
        
        // Make API request
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ symbol: symbol })
        })
        .then(response => response.json())
        .then(data => {
            loadingElement.classList.add('hidden');
            
            if (data.error) {
                showError(data.error);
                return;
            }
            
            displayResults(data);
        })
        .catch(error => {
            loadingElement.classList.add('hidden');
            showError('An error occurred: ' + error.message);
        });
    });
    
    function showError(message) {
        errorElement.classList.remove('hidden');
        errorElement.querySelector('.error-message').textContent = message;
    }

    // Number counter animation
    function animateCounter(element, target, prefix = "₹", duration = 1500) {
        let start = 0;
        const increment = target / (duration / 30); // update every ~30ms
        const interval = setInterval(() => {
            start += increment;
            if (start >= target) {
                start = target;
                clearInterval(interval);
            }
            element.textContent = prefix + Math.round(start).toLocaleString();
        }, 30);
    }
    
    function displayResults(data) {
        // Animate stock info numbers
        document.getElementById('stockName').textContent = data.symbol;
        animateCounter(document.getElementById('currentPrice'), data.current_price);
        animateCounter(document.getElementById('nextPrediction'), data.next_day_prediction);
        animateCounter(document.getElementById('modelAccuracy'), data.accuracy, "", 1000);

        // Smooth chart reveal
        const ctx = document.getElementById('stockChart').getContext('2d');
        const canvas = document.getElementById('stockChart');
        canvas.style.opacity = 0; // start hidden
        canvas.style.transition = "opacity 1s ease-in-out";
        
        if (stockChart) {
            stockChart.destroy();
        }
        
        stockChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: [
                    {
                        label: 'Actual Price',
                        data: data.actual,
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.3 // smooth line
                    },
                    {
                        label: 'Predicted Price',
                        data: data.predicted,
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 1200,
                    easing: 'easeOutQuart'
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Date'
                        },
                        ticks: {
                            maxTicksLimit: 10,
                            autoSkip: true
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Price (₹)'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Stock Price Prediction',
                        font: {
                            size: 16
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    },
                    legend: {
                        position: 'top',
                    }
                }
            }
        });

        // Reveal chart smoothly
        setTimeout(() => {
            canvas.style.opacity = 1;
        }, 100);

        // Show results
        resultsElement.classList.remove('hidden');
    }
});
