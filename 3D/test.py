import sys
import os

print("--- DIAGNOSTYKA ŚCIEŻEK ---")
print(f"Uruchamiam z katalogu: {os.getcwd()}")
print(f"Lokalizacja pliku main.py: {os.path.dirname(os.path.abspath(__file__))}")
print("Python szuka modułów w:")
for p in sys.path:
    print(f" - {p}")
print("---------------------------")

# Tutaj spróbujmy ręcznie dodać bieżący folder, żeby wymusić działanie:
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)