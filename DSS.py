import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tkinter as tk
from tkinter import messagebox
from tkinter import font as tkfont
from tkinter import ttk
from PIL import Image, ImageTk  


np.random.seed(42)

data = pd.read_csv('global_air_pollution_data.csv')
data.columns = [col.strip() for col in data.columns]  

data['country_name'] = data['country_name'].astype(str)

numeric_columns = ['aqi_value', 'co_aqi_value', 'ozone_aqi_value', 'no2_aqi_value', 'pm2.5_aqi_value']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

features = ['city_name', 'aqi_value', 'aqi_category', 'co_aqi_value', 'ozone_aqi_value', 'no2_aqi_value', 'pm2.5_aqi_value']
X = data[features].copy()
y = data['country_name']

for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42)
}

fitted_models = {name: model.fit(X_train, y_train) for name, model in models.items()}

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, cm

evaluation_results = {}
for model_name, model in fitted_models.items():
    accuracy, precision, recall, f1, cm = evaluate_model(model, X_test, y_test)
    evaluation_results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def update_cities(*args):
    selected_country = country_combobox.get()
    cities = sorted(data[data['country_name'] == selected_country]['city_name'].unique())
    city_combobox['values'] = cities
    city_combobox.set('')

def plot_data_for_city():
    city_name = city_combobox.get().strip()
    city_data = data[data['city_name'].str.lower() == city_name.lower()]
    
    if city_data.empty:
        messagebox.showerror("Error", "City not found!")
        return
    
    metrics = ['aqi_value', 'co_aqi_value', 'ozone_aqi_value', 'no2_aqi_value', 'pm2.5_aqi_value']
    metric_names = ['AQI', 'CO AQI', 'Ozone AQI', 'NO2 AQI', 'PM2.5 AQI']
    values = city_data[metrics].values.flatten()
    
    aqi = city_data['aqi_value'].values[0]
    air_quality = 'Good' if aqi <= 50 else 'Not Good'
    
    highest_pollutant = metric_names[np.argmax(values)]
    
    plt.figure(figsize=(10, 6))
    plt.bar(metric_names, values, color='skyblue')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title(f'Air Quality Metrics for {city_name}')
    
    plt.text(0.5, -0.4, f'Air Quality: {air_quality}', ha='center', fontsize=12, color='red', transform=plt.gca().transAxes)
    plt.text(0.5, -0.15, f'Highest Pollutant: {highest_pollutant}', ha='center', fontsize=12, color='red', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()

def open_feedback_dialog():
    def submit_feedback():
        feedback = feedback_text.get("1.0", tk.END).strip()
        if feedback:
            with open("feedback.txt", "a") as file:
                file.write(feedback + "\n")
            messagebox.showinfo("Feedback", "Thank you for your feedback!")
            feedback_dialog.destroy()
        else:
            messagebox.showwarning("Feedback", "Please enter your feedback before submitting.")
    
    feedback_dialog = tk.Toplevel(root)
    feedback_dialog.title("Feedback")
    feedback_dialog.configure(bg='light blue')
    feedback_dialog.geometry("400x300")
    
    tk.Label(feedback_dialog, text="Please provide your feedback:", font=custom_font, bg='light blue').pack(pady=10)
    feedback_text = tk.Text(feedback_dialog, width=40, height=10, font=custom_font)
    feedback_text.pack(pady=10)
    submit_button = tk.Button(feedback_dialog, text="Submit", command=submit_feedback, font=custom_font, bg='light blue')
    submit_button.pack(pady=10)

root = tk.Tk()
root.title("Air Quality Metrics")
root.state('zoomed')  
bg_image = Image.open("background.png")
bg_image = bg_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor="nw")

custom_font = tkfont.Font(family="Arial", size=14)

frame = tk.Frame(canvas, bg='light blue')
canvas.create_window(root.winfo_screenwidth()//2, root.winfo_screenheight()//2, window=frame, anchor="c")

tk.Label(frame, text="Select Country:", font=custom_font, bg='light blue').grid(row=0, column=0, padx=10, pady=10)
country_combobox = ttk.Combobox(frame, font=custom_font)
country_combobox['values'] = sorted(data['country_name'].unique())
country_combobox.grid(row=0, column=1, padx=10, pady=10)
country_combobox.bind('<<ComboboxSelected>>', update_cities)

tk.Label(frame, text="Select City:", font=custom_font, bg='light blue').grid(row=1, column=0, padx=10, pady=10)
city_combobox = ttk.Combobox(frame, font=custom_font)
city_combobox.grid(row=1, column=1, padx=10, pady=10)

plot_button = tk.Button(frame, text="Plot Data", command=plot_data_for_city, font=custom_font, bg='light blue')
plot_button.grid(row=2, column=0, columnspan=2, pady=10)

feedback_button = tk.Button(frame, text="Provide Feedback", command=open_feedback_dialog, font=custom_font, bg='light blue')
feedback_button.grid(row=3, column=0, columnspan=2, pady=10)

root.mainloop()
