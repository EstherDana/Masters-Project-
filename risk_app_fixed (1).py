# risk_app_fixed.py
# Project Risk Level Prediction System - 3 Model Comparison (FIXED)

import tkinter as tk
from tkinter import ttk, messagebox
from collections import Counter

# ============================================================
# LOGISTIC REGRESSION MODEL
# ============================================================

class LogisticRegressionModel:
    def __init__(self):
        self.coefficients = {
            'complexity': 0.45,
            'team_experience': 0.38,
            'org_maturity': 0.35,
            'turnover': 0.32,
            'timeline': 0.28,
            'success_rate': -0.25,
            'budget': 0.22,
            'integration': 0.20,
            'market_volatility': 0.18,
            'resource_availability': -0.18,
            'stakeholders': 0.15,
            'stability': -0.12,
            'sponsorship': -0.10,
        }
        self.intercept = -2.5
    
    def calculate_score(self, inputs):
        score = self.intercept
        for key, coef in self.coefficients.items():
            if key in inputs:
                value = inputs[key]
                # Ensure value is a number
                if isinstance(value, str):
                    try:
                        value = float(value)
                    except:
                        value = 0
                score += coef * value
        return max(-3, min(3, score))
    
    def predict(self, inputs):
        score = self.calculate_score(inputs)
        
        if score < -1.5:
            risk_level = "Low"
            probabilities = [0.75, 0.18, 0.05, 0.02]
        elif score < -0.5:
            risk_level = "Medium"
            probabilities = [0.20, 0.60, 0.15, 0.05]
        elif score < 0.5:
            risk_level = "High"
            probabilities = [0.05, 0.20, 0.60, 0.15]
        else:
            risk_level = "Critical"
            probabilities = [0.02, 0.08, 0.25, 0.65]
        
        return risk_level, probabilities, (score + 3) * 16.67


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

class RandomForestModel:
    def __init__(self):
        self.weights = {
            'complexity': 0.064,
            'org_maturity': 0.048,
            'team_experience': 0.045,
            'turnover': 0.044,
            'timeline': 0.035,
            'success_rate': 0.031,
            'budget': 0.029,
            'integration': 0.029,
            'market_volatility': 0.028,
            'resource_availability': 0.028,
            'stakeholders': 0.025,
            'stability': 0.022,
            'sponsorship': 0.018,
        }
    
    def calculate_score(self, inputs):
        score = 0.0
        for key, weight in self.weights.items():
            if key in inputs:
                value = inputs[key]
                if isinstance(value, str):
                    try:
                        value = float(value)
                    except:
                        value = 0
                score += weight * value * 100
        return min(100, max(0, score))
    
    def predict(self, inputs):
        score = self.calculate_score(inputs)
        
        if score < 25:
            risk_level = "Low"
            probabilities = [0.65, 0.25, 0.07, 0.03]
        elif score < 45:
            risk_level = "Medium"
            probabilities = [0.25, 0.55, 0.15, 0.05]
        elif score < 70:
            risk_level = "High"
            probabilities = [0.10, 0.30, 0.50, 0.10]
        else:
            risk_level = "Critical"
            probabilities = [0.05, 0.15, 0.30, 0.50]
        
        return risk_level, probabilities, score


# ============================================================
# XGBOOST MODEL
# ============================================================

class XGBoostModel:
    def __init__(self):
        self.weights = {
            'complexity': 0.031,
            'team_experience': 0.030,
            'org_maturity': 0.030,
            'tech_familiarity': 0.026,
            'risk_maturity': 0.025,
            'stakeholder_availability': 0.023,
            'project_manager_exp': 0.021,
            'stakeholder_engagement': 0.020,
            'timeline': 0.019,
            'turnover': 0.019,
            'integration': 0.018,
            'market_volatility': 0.017,
            'budget': 0.016,
            'stability': 0.015,
        }
    
    def calculate_score(self, inputs):
        score = 0.0
        for key, weight in self.weights.items():
            if key in inputs:
                val = inputs[key]
                if isinstance(val, str):
                    val = val.lower() if isinstance(val, str) else val
                
                if key == 'tech_familiarity':
                    tech_val = inputs.get('tech_familiarity', 'Familiar')
                    if isinstance(tech_val, str):
                        val = 1 if tech_val.lower() == 'new' else 0
                    else:
                        val = 0
                elif key == 'risk_maturity':
                    risk_val = inputs.get('risk_maturity', 'Basic')
                    if isinstance(risk_val, str):
                        val = 1 if risk_val.lower() == 'basic' else 0
                    else:
                        val = 0
                elif key == 'stakeholder_availability':
                    avail_map = {'good': 0, 'moderate': 0.33, 'limited': 0.67, 'poor': 1}
                    avail_val = inputs.get('stakeholder_availability', 'Moderate')
                    if isinstance(avail_val, str):
                        val = avail_map.get(avail_val.lower(), 0.5)
                    else:
                        val = 0.5
                elif key == 'project_manager_exp':
                    exp_map = {'senior pm': 0, 'mid-level pm': 0.5, 'junior pm': 1}
                    exp_val = inputs.get('project_manager_exp', 'Mid-level PM')
                    if isinstance(exp_val, str):
                        val = exp_map.get(exp_val.lower(), 0.5)
                    else:
                        val = 0.5
                elif key == 'stakeholder_engagement':
                    eng_map = {'excellent': 0, 'medium': 0.33, 'low': 0.67, 'poor': 1}
                    eng_val = inputs.get('stakeholder_engagement', 'Medium')
                    if isinstance(eng_val, str):
                        val = eng_map.get(eng_val.lower(), 0.5)
                    else:
                        val = 0.5
                else:
                    # Numeric value
                    if isinstance(val, str):
                        try:
                            val = float(val)
                        except:
                            val = 0
                
                score += weight * val * 100
        return min(100, max(0, score))
    
    def predict(self, inputs):
        score = self.calculate_score(inputs)
        
        if score < 22:
            risk_level = "Low"
            probabilities = [0.70, 0.22, 0.06, 0.02]
        elif score < 48:
            risk_level = "Medium"
            probabilities = [0.22, 0.58, 0.15, 0.05]
        elif score < 72:
            risk_level = "High"
            probabilities = [0.08, 0.25, 0.55, 0.12]
        else:
            risk_level = "Critical"
            probabilities = [0.03, 0.12, 0.28, 0.57]
        
        return risk_level, probabilities, score


# ============================================================
# MAIN APPLICATION
# ============================================================

class RiskPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project Risk Level Prediction - 3 Model Comparison")
        self.root.geometry("1100x850")
        self.root.configure(bg='#f0f0f0')
        
        self.models = {
            'Logistic Regression': LogisticRegressionModel(),
            'Random Forest': RandomForestModel(),
            'XGBoost': XGBoostModel()
        }
        
        self.risk_colors = {'Low': '#4CAF50', 'Medium': '#FFC107', 
                            'High': '#FF9800', 'Critical': '#f44336'}
        
        self.setup_ui()
        self.set_default_values()
    
    def setup_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#2196F3', height=80)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="Project Risk Level Prediction System",
                               font=('Arial', 18, 'bold'), bg='#2196F3', fg='white')
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="Comparing: Logistic Regression | Random Forest | XGBoost",
                                  font=('Arial', 11), bg='#2196F3', fg='#e0e0e0')
        subtitle_label.pack()
        
        # Main container with scroll
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill='both', expand=True, padx=15, pady=10)
        
        canvas = tk.Canvas(main_container, bg='#f0f0f0', highlightthickness=0)
        scrollbar = tk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        
        canvas.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)
        
        self.scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Input form
        self.inputs = {}
        row = 0
        
        # Input sections
        row = self.create_section("Basic Project Information", row)
        row = self.create_input("Project Name:", "project_name", row, default="New Project")
        row = self.create_input("Team Size (2-50):", "team_size", row, default="12")
        row = self.create_input("Project Budget (USD):", "budget", row, default="1000000")
        row = self.create_input("Timeline (months):", "timeline", row, default="12")
        row = self.create_input("Complexity Score (1-10):", "complexity", row, default="5")
        row = self.create_input("Stakeholder Count:", "stakeholders", row, default="10")
        
        row = self.create_section("Team Information", row)
        row = self.create_dropdown("Team Experience Level:", "team_experience", row,
                                   ['Junior', 'Mixed', 'Senior', 'Expert'], default="Mixed")
        row = self.create_input("Team Turnover Rate (0-1):", "turnover", row, default="0.2")
        row = self.create_dropdown("Project Manager Experience:", "project_manager_exp", row,
                                   ['Junior PM', 'Mid-level PM', 'Senior PM'], default="Mid-level PM")
        
        row = self.create_section("Project Management", row)
        row = self.create_dropdown("Requirement Stability:", "stability", row,
                                   ['Volatile', 'Moderate', 'Stable'], default="Moderate")
        row = self.create_dropdown("Priority Level:", "priority", row,
                                   ['Low', 'Medium', 'High', 'Critical'], default="Medium")
        
        row = self.create_section("Organizational Factors", row)
        row = self.create_dropdown("Org Process Maturity:", "org_maturity", row,
                                   ['Ad-hoc', 'Managed', 'Defined', 'Optimizing'], default="Managed")
        row = self.create_dropdown("Executive Sponsorship:", "sponsorship", row,
                                   ['Weak', 'Moderate', 'Strong'], default="Moderate")
        row = self.create_input("Previous Delivery Success Rate (0-1):", "success_rate", row, default="0.75")
        row = self.create_dropdown("Risk Management Maturity:", "risk_maturity", row,
                                   ['Basic', 'Formal'], default="Basic")
        
        row = self.create_section("Stakeholder Information", row)
        row = self.create_dropdown("Stakeholder Engagement:", "stakeholder_engagement", row,
                                   ['Poor', 'Low', 'Medium', 'Excellent'], default="Medium")
        row = self.create_dropdown("Key Stakeholder Availability:", "stakeholder_availability", row,
                                   ['Poor', 'Limited', 'Moderate', 'Good'], default="Moderate")
        
        row = self.create_section("Technical Factors", row)
        row = self.create_input("Integration Complexity (1-10):", "integration", row, default="5")
        row = self.create_dropdown("Technology Familiarity:", "tech_familiarity", row,
                                   ['Familiar', 'New'], default="Familiar")
        
        row = self.create_section("External Factors", row)
        row = self.create_input("Market Volatility (0-1):", "market_volatility", row, default="0.5")
        row = self.create_input("Resource Availability (0-1):", "resource_availability", row, default="0.7")
        
        # Buttons
        button_frame = tk.Frame(self.scrollable_frame, bg='#f0f0f0')
        button_frame.grid(row=row, column=0, columnspan=2, pady=20)
        
        predict_btn = tk.Button(button_frame, text="PREDICT WITH ALL MODELS", 
                                font=('Arial', 14, 'bold'), bg='#2196F3', fg='white',
                                padx=30, pady=10, command=self.predict_all)
        predict_btn.pack(side='left', padx=10)
        
        clear_btn = tk.Button(button_frame, text="CLEAR FORM", 
                              font=('Arial', 12), bg='#757575', fg='white',
                              padx=20, pady=10, command=self.clear_form)
        clear_btn.pack(side='left', padx=10)
        
        # Results area
        self.results_frame = tk.Frame(self.scrollable_frame, bg='#f0f0f0')
        self.results_frame.grid(row=row+1, column=0, columnspan=2, pady=20, sticky='ew')
        
        self.model_results = {}
        colors = ['#E3F2FD', '#E8F5E9', '#FFF3E0']
        
        for i, (model_name, _) in enumerate(self.models.items()):
            col_frame = tk.Frame(self.results_frame, bg=colors[i], relief='ridge', bd=2)
            col_frame.pack(side='left', expand=True, fill='both', padx=5, pady=5)
            
            header = tk.Label(col_frame, text=model_name, font=('Arial', 14, 'bold'),
                              bg=colors[i], fg='#333')
            header.pack(pady=10)
            
            risk_label = tk.Label(col_frame, text="--", font=('Arial', 16, 'bold'),
                                  bg=colors[i], fg='#333')
            risk_label.pack(pady=5)
            
            score_label = tk.Label(col_frame, text="Score: --", font=('Arial', 10),
                                   bg=colors[i], fg='#666')
            score_label.pack()
            
            prob_frame = tk.Frame(col_frame, bg=colors[i])
            prob_frame.pack(pady=10, padx=10, fill='x')
            
            progress_bars = {}
            for risk in ['Low', 'Medium', 'High', 'Critical']:
                frame = tk.Frame(prob_frame, bg=colors[i])
                frame.pack(fill='x', pady=2)
                
                label = tk.Label(frame, text=f"{risk}:", width=8, anchor='w', 
                                 font=('Arial', 9), bg=colors[i])
                label.pack(side='left')
                
                progress = ttk.Progressbar(frame, length=120, mode='determinate')
                progress.pack(side='left', padx=5)
                
                value_label = tk.Label(frame, text="0%", width=6, anchor='w',
                                        font=('Arial', 9), bg=colors[i])
                value_label.pack(side='left')
                
                progress_bars[risk] = (progress, value_label)
            
            self.model_results[model_name] = {
                'risk_label': risk_label,
                'score_label': score_label,
                'progress_bars': progress_bars
            }
        
        # Ensemble summary
        ensemble_frame = tk.Frame(self.results_frame, bg='#E8EAF6', relief='ridge', bd=2)
        ensemble_frame.pack(side='bottom', fill='x', padx=5, pady=10)
        
        ensemble_title = tk.Label(ensemble_frame, text="ENSEMBLE (Majority Vote)", 
                                   font=('Arial', 12, 'bold'), bg='#E8EAF6', fg='#333')
        ensemble_title.pack(pady=5)
        
        self.ensemble_label = tk.Label(ensemble_frame, text="--", font=('Arial', 14, 'bold'),
                                        bg='#E8EAF6')
        self.ensemble_label.pack(pady=5)
    
    def create_section(self, text, row):
        header = tk.Label(self.scrollable_frame, text=text, font=('Arial', 12, 'bold'),
                          bg='#e0e0e0', fg='#333', anchor='w', padx=10)
        header.grid(row=row, column=0, columnspan=2, sticky='ew', pady=(15, 5), padx=5)
        header.configure(relief='ridge')
        return row + 1
    
    def create_input(self, label_text, key, row, default=""):
        label = tk.Label(self.scrollable_frame, text=label_text, font=('Arial', 10),
                         bg='#f0f0f0', anchor='e', width=30)
        label.grid(row=row, column=0, sticky='e', padx=(10, 5), pady=5)
        
        var = tk.StringVar()
        entry = tk.Entry(self.scrollable_frame, textvariable=var, width=25)
        entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        
        if default:
            var.set(default)
        
        self.inputs[key] = var
        return row + 1
    
    def create_dropdown(self, label_text, key, row, options, default=""):
        label = tk.Label(self.scrollable_frame, text=label_text, font=('Arial', 10),
                         bg='#f0f0f0', anchor='e', width=30)
        label.grid(row=row, column=0, sticky='e', padx=(10, 5), pady=5)
        
        var = tk.StringVar()
        dropdown = ttk.Combobox(self.scrollable_frame, textvariable=var, values=options, width=23)
        dropdown.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        
        if default:
            var.set(default)
        elif options:
            var.set(options[0])
        
        self.inputs[key] = var
        return row + 1
    
    def set_default_values(self):
        defaults = {
            'team_size': '12',
            'budget': '1000000',
            'timeline': '12',
            'complexity': '5',
            'stakeholders': '10',
            'turnover': '0.2',
            'success_rate': '0.75',
            'integration': '5',
            'market_volatility': '0.5',
            'resource_availability': '0.7',
        }
        
        for key, value in defaults.items():
            if key in self.inputs:
                self.inputs[key].set(value)
    
    def get_inputs(self):
        """Get all input values and convert to proper types"""
        result = {}
        
        # First get all raw values
        for key, var in self.inputs.items():
            value = var.get().strip()
            if not value:
                value = "0"
            result[key] = value
        
        # Convert numeric fields to floats
        numeric_fields = ['team_size', 'budget', 'timeline', 'complexity', 
                         'stakeholders', 'turnover', 'success_rate', 
                         'integration', 'market_volatility', 'resource_availability']
        
        for field in numeric_fields:
            if field in result:
                try:
                    result[field] = float(result[field])
                except ValueError:
                    result[field] = 0.0
        
        # Normalize values to 0-1 range for models
        if 'team_size' in result:
            result['team_size'] = min(1.0, result['team_size'] / 50)
        if 'budget' in result:
            result['budget'] = min(1.0, result['budget'] / 5000000)
        if 'timeline' in result:
            result['timeline'] = min(1.0, result['timeline'] / 36)
        if 'complexity' in result:
            result['complexity'] = result['complexity'] / 10
        if 'stakeholders' in result:
            result['stakeholders'] = min(1.0, result['stakeholders'] / 30)
        if 'integration' in result:
            result['integration'] = result['integration'] / 10
        if 'turnover' in result:
            result['turnover'] = min(1.0, max(0.0, result['turnover']))
        if 'success_rate' in result:
            result['success_rate'] = min(1.0, max(0.0, result['success_rate']))
        if 'market_volatility' in result:
            result['market_volatility'] = min(1.0, max(0.0, result['market_volatility']))
        if 'resource_availability' in result:
            result['resource_availability'] = min(1.0, max(0.0, result['resource_availability']))
        
        # Categorical values - keep as strings for models to process
        categorical_fields = ['team_experience', 'stability', 'org_maturity', 
                              'sponsorship', 'stakeholder_engagement', 
                              'stakeholder_availability', 'tech_familiarity',
                              'risk_maturity', 'project_manager_exp']
        
        for field in categorical_fields:
            if field in result and field in self.inputs:
                result[field] = self.inputs[field].get()
        
        return result
    
    def predict_all(self):
        try:
            inputs = self.get_inputs()
            
            predictions = {}
            all_risk_levels = []
            
            for model_name, model in self.models.items():
                risk_level, probabilities, score = model.predict(inputs)
                predictions[model_name] = (risk_level, probabilities, score)
                all_risk_levels.append(risk_level)
            
            for model_name, (risk_level, probabilities, score) in predictions.items():
                results = self.model_results[model_name]
                results['risk_label'].config(text=risk_level, fg=self.risk_colors[risk_level])
                results['score_label'].config(text=f"Score: {score:.1f}/100")
                
                for risk, prob in zip(['Low', 'Medium', 'High', 'Critical'], probabilities):
                    progress, label = results['progress_bars'][risk]
                    percent = prob * 100
                    progress['value'] = percent
                    label.config(text=f"{percent:.1f}%")
            
            vote_counts = Counter(all_risk_levels)
            ensemble_risk = vote_counts.most_common(1)[0][0]
            
            self.ensemble_label.config(text=f"ENSEMBLE PREDICTION: {ensemble_risk}", 
                                        fg=self.risk_colors[ensemble_risk])
            
            if 'High' in all_risk_levels or 'Critical' in all_risk_levels:
                high_models = [m for m, (r, _, _) in predictions.items() if r in ['High', 'Critical']]
                messagebox.showwarning("High Risk Alert", 
                    f"Models predicting HIGH/CRITICAL risk: {', '.join(high_models)}\n"
                    f"Ensemble: {ensemble_risk}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def clear_form(self):
        for key, var in self.inputs.items():
            var.set("")
        
        for model_name, results in self.model_results.items():
            results['risk_label'].config(text="--", fg='#333')
            results['score_label'].config(text="Score: --")
            for risk, (progress, label) in results['progress_bars'].items():
                progress['value'] = 0
                label.config(text="0%")
        
        self.ensemble_label.config(text="--")
        self.set_default_values()


# ============================================================
# MAIN
# ============================================================

def main():
    root = tk.Tk()
    app = RiskPredictionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()