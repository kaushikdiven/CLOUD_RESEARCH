# Comparative Evaluation of GA, ACO, and RL for Cloud-Aware Task Scheduling in Smart Home and IoT Environments

## 📄 Abstract

This repository provides the source code, datasets (if available), and documentation for the comparative study of **Genetic Algorithm (GA)**, **Ant Colony Optimization (ACO)**, and **Reinforcement Learning (RL)** techniques in cloud-aware task scheduling. The project evaluates these algorithms under identical IoT task loads and compares their effectiveness in terms of **energy efficiency**, **task completion (QoS)**, and **computational overhead**.

## 🧠 Objective

To experimentally analyze and compare GA, ACO, and RL in smart environments enabled by cloud computing, focusing on:

* Reducing energy consumption
* Improving Quality of Service (QoS)
* Managing computational cost

## 🏗️ Architecture

```
IoT Devices --> Task Scheduler (GA/ACO/RL) --> Cloud Resources
                                  |
                               Performance Metrics
```

Each algorithm schedules tasks on cloud infrastructure with varying priorities, deadlines, and energy budgets.

## 🛠️ Technologies & Libraries

| Algorithm | Language | Libraries/Repos Used                                                                                                                      |
| --------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| GA        | Python   | Ga ([https://github.com/DomenicoDeFelice/genetic-algorithm-in-python](https://github.com/DomenicoDeFelice/genetic-algorithm-in-python))   |
| ACO       | Python   | Ant Colony Optimization ([https://github.com/Akavall/AntColonyOptimization](https://github.com/Akavall/AntColonyOptimization))            |
| RL        | Python   | RL Algorithms ([https://github.com/1Kaustubh122/RL\_Algorithms](https://github.com/1Kaustubh122/RL_Algorithms)),                          |

## 📊 Results Summary

| Algorithm | Energy Efficiency (%) | QoS (Task Completion %) | Overhead (ms/task) |
| --------- | --------------------- | ----------------------- | ------------------ |
| GA        | 68.5                  | 84.2                    | 14.3               |
| ACO       | 72.1                  | 79.8                    | 18.7               |
| RL        | 70.4                  | 88.5                    | 23.9               |

## 📈 Key Insights

* **GA** is fast and low on overhead but lacks adaptability.
* **ACO** provides the best energy savings for low task volumes.
* **RL** achieves the highest QoS but with high computational cost.
* Cloud computing amplifies the benefits of all three by offering scalability and offloading computation.


## 🚀 How to Run

1. Clone the repository:

   ```
   git clone https://github.com/kaushikdiven/CLOUD_RESEARCH.git
   cd cloud-task-scheduling
   ```

2. Set up a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

## 🔍 Future Work

* Integration with edge/5G systems
* Hybrid models combining GA, ACO, and RL
* Multi-objective optimization in real deployments

## 📚 Citation

If you use this repository in your research, please cite:

```
Diven, Dr. Naween Kumar. "Comparative Evaluation of GA, ACO, and RL for Cloud-Aware Task Scheduling in Smart Home and IoT Environments." Bennett University, India.
```

## 🤝 Contributors

* **Dr. Naween Kumar** – Supervisor
* **Diven** – Researcher & Developer
