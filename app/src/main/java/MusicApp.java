import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpExchange;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.*;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.feature.StringIndexerModel;

import java.io.IOException;
import java.io.OutputStream;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

public class MusicApp {

    private static String[] GENRES;
    private static SparkSession spark;
    private static PipelineModel model;

    public static void main(String[] args) throws Exception {
        System.out.println("Initializing Spark Session...");
        spark = SparkSession.builder()
                .appName("LyricsWebUI")
                .master("local[*]")
                .getOrCreate();

        spark.sparkContext().setLogLevel("WARN");

        System.out.println("Loading trained model...");
        model = PipelineModel.load("model/trained_lyrics_model");

        StringIndexerModel indexer = (StringIndexerModel) model.stages()[0];
        GENRES = indexer.labels();

        HttpServer server = HttpServer.create(new InetSocketAddress(5000), 0);
        server.createContext("/", new IndexHandler());
        server.createContext("/predict", new PredictHandler());
        server.setExecutor(null);
        server.start();

        System.out.println("Server is running on http://127.0.0.1:5000");
    }

    static class IndexHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            String html = "<!DOCTYPE html>\n" +
                    "<html lang=\"en\">\n" +
                    "<head>\n" +
                    "    <meta charset=\"UTF-8\">\n" +
                    "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n" +
                    "    <title>Music Genre Classifier</title>\n" +
                    "    <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n" +
                    "    <link href=\"https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap\" rel=\"stylesheet\">\n" +
                    "    <style>\n" +
                    "        :root { --bg: #121212; --surface: #1e1e1e; --primary: #1db954; --text: #ffffff; --text-muted: #b3b3b3; }\n" +
                    "        body { font-family: 'Poppins', sans-serif; background: var(--bg); color: var(--text); margin: 0; display: flex; justify-content: center; padding: 40px 20px; }\n" +
                    "        .container { background: var(--surface); max-width: 800px; width: 100%; padding: 40px; border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,0.6); }\n" +
                    "        h2 { margin-top: 0; text-align: center; color: var(--text); letter-spacing: 1px; font-size: 2em; }\n" +
                    "        h2 span { color: var(--primary); }\n" +
                    "        p.subtitle { text-align: center; color: var(--text-muted); font-size: 0.9em; margin-bottom: 25px; }\n" +
                    "        textarea { width: 100%; box-sizing: border-box; background: #282828; color: var(--text); border: 1px solid #333; border-radius: 8px; padding: 15px; font-family: inherit; resize: vertical; outline: none; transition: border 0.3s; }\n" +
                    "        textarea:focus { border-color: var(--primary); }\n" +
                    "        button { display: block; width: 100%; background: var(--primary); color: #000; font-weight: 600; font-size: 16px; font-family: inherit; border: none; padding: 15px; margin-top: 20px; border-radius: 50px; cursor: pointer; transition: transform 0.2s, background 0.2s; }\n" +
                    "        button:hover { background: #1ed760; transform: scale(1.02); }\n" +
                    "        button:active { transform: scale(0.98); }\n" +
                    "        .chart-container { margin-top: 40px; position: relative; height: 350px; width: 100%; display: none; }\n" +
                    "    </style>\n" +
                    "</head>\n" +
                    "<body>\n" +
                    "    <div class=\"container\">\n" +
                    "        <h2>🎵 AI Music <span>Classifier</span></h2>\n" +
                    "        <p class=\"subtitle\">Paste song lyrics below to predict the genre</p>\n" +
                    "        <textarea id=\"lyrics\" rows=\"8\" placeholder=\"I hear the train a comin'...\"></textarea>\n" +
                    "        <button id=\"btn\" onclick=\"classify()\">Analyze Genre</button>\n" +
                    "        <div class=\"chart-container\" id=\"chartBox\"><canvas id=\"chart\"></canvas></div>\n" +
                    "    </div>\n" +
                    "    <script>\n" +
                    "        let c = null;\n" +
                    "        Chart.defaults.color = '#b3b3b3';\n" +
                    "        Chart.defaults.font.family = 'Poppins';\n" +
                    "        function classify() {\n" +
                    "            const btn = document.getElementById('btn');\n" +
                    "            const lyrics = document.getElementById('lyrics').value;\n" +
                    "            if(!lyrics.trim()) return alert('Please enter some lyrics first!');\n" +
                    "            \n" +
                    "            btn.innerText = 'Analyzing...';\n" +
                    "            btn.style.opacity = '0.8';\n" +
                    "            \n" +
                    "            fetch('/predict', {\n" +
                    "                method: 'POST',\n" +
                    "                body: 'lyrics=' + encodeURIComponent(lyrics)\n" +
                    "            })\n" +
                    "            .then(r => r.json())\n" +
                    "            .then(d => {\n" +
                    "                btn.innerText = 'Analyze Genre';\n" +
                    "                btn.style.opacity = '1';\n" +
                    "                document.getElementById('chartBox').style.display = 'block';\n" +
                    "                if(c) c.destroy();\n" +
                    "                \n" +
                    "                const ctx = document.getElementById('chart').getContext('2d');\n" +
                    "                const gradient = ctx.createLinearGradient(0, 0, 0, 350);\n" +
                    "                gradient.addColorStop(0, '#1db954');\n" +
                    "                gradient.addColorStop(1, 'rgba(29, 185, 84, 0.1)');\n" +
                    "                \n" +
                    "                c = new Chart(ctx, {\n" +
                    "                    type: 'bar',\n" +
                    "                    data: {\n" +
                    "                        labels: Object.keys(d).map(l => l.toUpperCase()),\n" +
                    "                        datasets: [{\n" +
                    "                            label: 'Probability (%)',\n" +
                    "                            data: Object.values(d),\n" +
                    "                            backgroundColor: gradient,\n" +
                    "                            borderRadius: 6,\n" +
                    "                            borderWidth: 0\n" +
                    "                        }]\n" +
                    "                    },\n" +
                    "                    options: {\n" +
                    "                        responsive: true,\n" +
                    "                        maintainAspectRatio: false,\n" +
                    "                        plugins: { legend: { display: false } },\n" +
                    "                        scales: {\n" +
                    "                            y: { max: 100, grid: { color: '#333' } },\n" +
                    "                            x: { grid: { display: false } }\n" +
                    "                        }\n" +
                    "                    }\n" +
                    "                });\n" +
                    "            }).catch(e => {\n" +
                    "                btn.innerText = 'Analyze Genre';\n" +
                    "                btn.style.opacity = '1';\n" +
                    "                alert('Error analyzing lyrics.');\n" +
                    "            });\n" +
                    "        }\n" +
                    "    </script>\n" +
                    "</body>\n" +
                    "</html>";

            exchange.getResponseHeaders().set("Content-Type", "text/html; charset=UTF-8");
            exchange.sendResponseHeaders(200, html.getBytes().length);
            OutputStream os = exchange.getResponseBody();
            os.write(html.getBytes());
            os.close();
        }
    }

    static class PredictHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if ("POST".equals(exchange.getRequestMethod())) {
                InputStream is = exchange.getRequestBody();
                String body = new String(is.readAllBytes(), StandardCharsets.UTF_8);
                String lyrics = URLDecoder.decode(body.replace("lyrics=", ""), StandardCharsets.UTF_8);

                StructType schema = new StructType(new StructField[]{
                        new StructField("lyrics", DataTypes.StringType, false, Metadata.empty())
                });
                Dataset<Row> df = spark.createDataFrame(Arrays.asList(RowFactory.create(lyrics)), schema);

                Dataset<Row> prediction = model.transform(df);
                Vector probVector = (Vector) prediction.select("probability").first().get(0);

                StringBuilder json = new StringBuilder("{");
                for (int i = 0; i < GENRES.length; i++) {
                    double percentage = probVector.apply(i) * 100.0;
                    json.append("\"").append(GENRES[i]).append("\":").append(String.format("%.2f", percentage));
                    if (i < GENRES.length - 1) json.append(",");
                }
                json.append("}");

                exchange.getResponseHeaders().set("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, json.toString().getBytes().length);
                OutputStream os = exchange.getResponseBody();
                os.write(json.toString().getBytes());
                os.close();
            }
        }
    }
}