<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Price Filter</title>
    <style>
        /* General Styles */
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            text-align: center;
            margin-bottom: 20px;
        }

        .logo {
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
        }

        nav ul {
            list-style: none;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            gap: 15px;
        }

        nav a {
            text-decoration: none;
            color: #333;
            font-size: 16px;
            transition: color 0.3s;
        }

        nav a:hover {
            color: #007BFF;
        }

        form {
            text-align: center;
            margin: 30px 0;
        }

        form label {
            font-size: 20px; /* Increased font size for the label */
            font-weight: bold;
            color: #2c3e50;
        }

        form input {
            padding: 10px;
            width: 80%;
            max-width: 300px;
            margin-right: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }

        form button {
            padding: 10px 20px;
            background-color: #007BFF;
            color: #fff;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }

        form button:hover {
            background-color: #0056b3;
        }

        .results {
            text-align: center;
            margin-top: 30px;
        }

        .results h3 {
            color: #2980b9;
            font-size: 1.5em;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }

        table th, table td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: center;
        }

        table th {
            background-color: #007BFF;
            color: #fff;
        }

        /* Responsive Styles */
        @media (max-width: 768px) {
            nav ul {
                flex-direction: column;
                align-items: center;
            }

            form input, form button {
                width: 100%;
                max-width: none;
            }

            table th, table td {
                font-size: 14px;
            }
        }

        footer {
            text-align: center;
            padding: 10px;
            margin-top: 20px;
            border-top: 2px solid #ddd;
            background-color: #2c3e50;
            color: #fff;
            border-radius: 0 0 10px 10px;
        }

        footer p {
            margin: 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <nav>
                <ul>
                    <li><a href="{{ url_for('homepage') }}">Home</a></li>
                    <li><a href="{{ url_for('about') }}">About</a></li>
                    <li><a href="{{ url_for('contact') }}">Contact</a></li>
                    <li><a href="{{ url_for('stocks') }}">Find Companies</a></li>
                </ul>
            </nav>
            <h1 class="logo">Predict the Future of Your Investments</h1>
        </header>

        <form method="POST" action="/stock_price">
            <label for="max_price">Enter the maximum stock price (INR):</label><br><br>
            <input type="text" id="max_price" name="max_price" placeholder="Enter max price" required>
            
            <button type="submit">Filter</button>
            <h3>Please wait this may take a while.....</h3>
        </form>

        {% if top_100 %}
        <div class="results">
            <h3>Top 100 Companies with Stock Price Below ₹{{ max_price }}:</h3>
            <table>
                <thead>
                    <tr>
                        <th>Company</th>
                        <th>Price (INR)</th>
                    </tr>
                </thead>
                <tbody>
                    {% for company, price in top_100 %}
                    <tr>
                        <td>{{ company }}</td>
                        <td>₹{{ "%.2f"|format(price) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
    </div>
</body>
</html>
