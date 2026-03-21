@echo off
echo Starting the Native Java Spark Server...

start /B spark-submit --class MusicApp --master local[*] app/target/MusicClassifier-1.0.jar

timeout /t 15 /nobreak > NUL

echo Launching browser...
start http://127.0.0.1:5000

echo.
echo Server is running! Press any key to shut it down when you're done.
pause > NUL

taskkill /IM java.exe /F > NUL