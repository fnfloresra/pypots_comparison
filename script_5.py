# Create VSCode setup guide and requirements files

vscode_setup_guide = '''
# ===============================================
# VSCode Setup Guide for PyPOTS Projects
# ===============================================

## Project Structure Setup

Create the following folder structure for your PyPOTS comparison project:

```
pypots_comparison_project/
│
├── environments/
│   ├── knn_env/
│   ├── brits_env/
│   └── saits_env/
│
├── src/
│   ├── knn_implementation.py
│   ├── brits_implementation.py
│   ├── saits_implementation.py
│   └── comparison_framework.py
│
├── data/
│   └── your_dataset.csv
│
├── results/
│   ├── plots/
│   └── reports/
│
├── notebooks/
│   ├── knn_analysis.ipynb
│   ├── brits_analysis.ipynb
│   └── saits_analysis.ipynb
│
├── requirements/
│   ├── knn_requirements.txt
│   ├── brits_requirements.txt
│   └── saits_requirements.txt
│
└── vscode_settings/
    ├── settings.json
    ├── launch.json
    └── tasks.json
```

## Step-by-Step VSCode Setup

### 1. Install Required VSCode Extensions

Install these extensions for optimal Python development:

- Python (Microsoft)
- Python Environment Manager
- Jupyter
- Python Docstring Generator
- Python Type Hint
- Python Test Explorer
- GitLens
- GitHub Copilot (optional)

### 2. Create Virtual Environments

Open terminal in VSCode (Ctrl+`) and run these commands:

```bash
# Navigate to your project directory
cd pypots_comparison_project

# Create virtual environments for each method
python -m venv environments/knn_env
python -m venv environments/brits_env  
python -m venv environments/saits_env

# Activate and install dependencies for KNN environment
# Windows:
environments\\knn_env\\Scripts\\activate
# macOS/Linux:
source environments/knn_env/bin/activate

pip install -r requirements/knn_requirements.txt

# Deactivate and repeat for BRITS
deactivate

# Windows:
environments\\brits_env\\Scripts\\activate
# macOS/Linux:
source environments/brits_env/bin/activate

pip install -r requirements/brits_requirements.txt

# Deactivate and repeat for SAITS
deactivate

# Windows:
environments\\saits_env\\Scripts\\activate
# macOS/Linux:
source environments/saits_env/bin/activate

pip install -r requirements/saits_requirements.txt

deactivate
```

### 3. Configure VSCode Settings

Create `.vscode/settings.json` in your project root:

```json
{
    "python.defaultInterpreterPath": "./environments/knn_env/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.envFile": "${workspaceFolder}/.env",
    "jupyter.jupyterServerType": "local",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "files.associations": {
        "*.py": "python"
    },
    "python.analysis.extraPaths": [
        "./src"
    ],
    "python.pythonPath": "./environments/knn_env/bin/python"
}
```

### 4. Create Launch Configurations

Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "KNN Analysis",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/knn_implementation.py",
            "python": "${workspaceFolder}/environments/knn_env/bin/python",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        },
        {
            "name": "BRITS Analysis",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/brits_implementation.py",
            "python": "${workspaceFolder}/environments/brits_env/bin/python",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        },
        {
            "name": "SAITS Analysis",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/saits_implementation.py",
            "python": "${workspaceFolder}/environments/saits_env/bin/python",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        },
        {
            "name": "Comprehensive Comparison",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/comparison_framework.py",
            "python": "${workspaceFolder}/environments/saits_env/bin/python",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        }
    ]
}
```

### 5. Create Tasks for Environment Management

Create `.vscode/tasks.json`:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Activate KNN Environment",
            "type": "shell",
            "command": "source",
            "args": ["environments/knn_env/bin/activate"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Activate BRITS Environment", 
            "type": "shell",
            "command": "source",
            "args": ["environments/brits_env/bin/activate"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Activate SAITS Environment",
            "type": "shell", 
            "command": "source",
            "args": ["environments/saits_env/bin/activate"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Run KNN Analysis",
            "type": "shell",
            "command": "environments/knn_env/bin/python",
            "args": ["src/knn_implementation.py"],
            "group": "build",
            "dependsOn": "Activate KNN Environment"
        },
        {
            "label": "Run BRITS Analysis",
            "type": "shell",
            "command": "environments/brits_env/bin/python", 
            "args": ["src/brits_implementation.py"],
            "group": "build",
            "dependsOn": "Activate BRITS Environment"
        },
        {
            "label": "Run SAITS Analysis",
            "type": "shell",
            "command": "environments/saits_env/bin/python",
            "args": ["src/saits_implementation.py"], 
            "group": "build",
            "dependsOn": "Activate SAITS Environment"
        }
    ]
}
```

## Working with Different Environments in VSCode

### Method 1: Using Command Palette
1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)
2. Type "Python: Select Interpreter"
3. Choose the environment you want to use:
   - `./environments/knn_env/bin/python` for KNN
   - `./environments/brits_env/bin/python` for BRITS
   - `./environments/saits_env/bin/python` for SAITS

### Method 2: Using Status Bar
1. Click on the Python version in the bottom status bar
2. Select the desired environment from the list

### Method 3: Using Tasks
1. Press `Ctrl+Shift+P` and type "Tasks: Run Task" 
2. Select the environment activation task you want

## Running Individual Methods

### For KNN Analysis:
1. Select KNN environment interpreter
2. Open `src/knn_implementation.py`
3. Press `F5` or use "Run Python File" 
4. Or use the "KNN Analysis" launch configuration

### For BRITS Analysis:
1. Select BRITS environment interpreter  
2. Open `src/brits_implementation.py`
3. Press `F5` or use "Run Python File"
4. Or use the "BRITS Analysis" launch configuration

### For SAITS Analysis:
1. Select SAITS environment interpreter
2. Open `src/saits_implementation.py` 
3. Press `F5` or use "Run Python File"
4. Or use the "SAITS Analysis" launch configuration

### For Comprehensive Comparison:
1. Select SAITS environment (most comprehensive)
2. Open `src/comparison_framework.py`
3. Use the "Comprehensive Comparison" launch configuration

## Debugging Setup

### Breakpoint Debugging:
1. Set breakpoints by clicking on line numbers
2. Use the appropriate launch configuration 
3. Press `F5` to start debugging
4. Use debugging controls:
   - `F10`: Step Over
   - `F11`: Step Into
   - `Shift+F11`: Step Out
   - `F5`: Continue

### Variable Inspection:
- Use the Variables panel during debugging
- Hover over variables to see values
- Use the Debug Console for interactive evaluation

## Jupyter Notebook Integration

### Setting up Notebooks:
1. Create notebooks in the `notebooks/` folder
2. When opening a notebook, select the appropriate kernel:
   - KNN: `environments/knn_env/bin/python`
   - BRITS: `environments/brits_env/bin/python` 
   - SAITS: `environments/saits_env/bin/python`

### Running Notebooks:
1. Open any `.ipynb` file
2. VSCode will automatically detect Jupyter
3. Select the appropriate Python kernel
4. Use `Shift+Enter` to run cells

## Git Integration

### Initial Setup:
```bash
git init
git add .
git commit -m "Initial PyPOTS comparison project setup"
```

### .gitignore File:
Create `.gitignore`:
```
# Virtual environments
environments/
*.venv
venv/
env/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebooks
.ipynb_checkpoints

# Data files (optional - depending on size)
data/*.csv
data/*.h5
data/*.hdf5

# Results
results/plots/*.png
results/reports/*.json
results/reports/*.txt

# IDE
.vscode/settings.json
.idea/

# OS
.DS_Store
Thumbs.db
```

## Troubleshooting Common Issues

### 1. Environment not detected:
- Reload VSCode window (`Ctrl+Shift+P` → "Developer: Reload Window")
- Check interpreter path in status bar
- Verify environment creation was successful

### 2. Import errors:
- Ensure PYTHONPATH includes src directory
- Check that all requirements are installed in active environment
- Verify environment is properly activated

### 3. PyTorch/CUDA issues:
- For BRITS and SAITS, ensure compatible PyTorch version
- Check CUDA availability: `torch.cuda.is_available()`
- Install CPU version if no GPU: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

### 4. Memory issues:
- Reduce batch size in model configurations
- Use smaller sequence lengths for testing
- Monitor system resources during training

### 5. Performance issues:
- Enable GPU usage if available
- Reduce model complexity for initial testing
- Use data sampling for quick iterations

## Best Practices

1. **Environment Management:**
   - Keep environments separate for each method
   - Document all dependencies in requirements files
   - Use environment-specific launch configurations

2. **Code Organization:**
   - Keep implementations in separate files
   - Use common interfaces for fair comparison
   - Document all hyperparameters and assumptions

3. **Data Management:**
   - Store original data in `data/` folder
   - Save processed data with clear naming
   - Version control preprocessing scripts

4. **Results Management:**
   - Save all results with timestamps
   - Include model parameters in result files
   - Create reproducible analysis notebooks

5. **Documentation:**
   - Comment complex algorithms
   - Document parameter choices
   - Keep README updated with setup instructions

This setup provides a comprehensive VSCode environment for comparing KNN, BRITS, and SAITS methods using PyPOTS, with proper isolation between environments and tools for effective development and analysis.
'''

print("VSCode Setup Guide Created!")
print("Length:", len(vscode_setup_guide))