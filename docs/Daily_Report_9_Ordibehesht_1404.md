
# 📝 Daily Report  
**📅 Date:** 9 Ordibehesht 1404  
**👤 Name:** Abolfazel Dehbozorgi  

## 🎯 Activities  

### GAN – Scheduler Fixes & Loss Function Debugging  

- ✅ **Fixed CosineAnnealingWarmRestarts Misuse**  
  **Before (Incorrect):**
  ```python
  optims_g, = CosineAnnealingWarmRestarts(optimizer=optim.Adam(generator.parameters(), lr=0.0002), T_0=10)
  ```
  **After (Correct):**
  ```python
  optims_g = optim.Adam(generator.parameters(), lr=0.0002)
  scheduler_g = CosineAnnealingWarmRestarts(optimizer=optims_g, T_0=10)
  ```
  > ⚠️ Always initialize the optimizer before passing it to the scheduler.

- 🧠 **Corrected Loss Function Configuration**  
  - Previously used `BCELoss` with `nn.Sigmoid` while using `autocast`, causing runtime errors.  
  - Switched to `BCEWithLogitsLoss`, which integrates `Sigmoid` internally and supports mixed precision.  
  - ✅ Apology for the mistake in the previous report — now rectified.

---

## 📊 Quick Results Snapshot (With Scheduler + Pruning)

| Epoch | D_Loss | MAE    | MSE    | RMSE   | R² Score |
|-------|--------|--------|--------|--------|----------|
| 0     | 0.517  | 0.0778 | 0.0147 | 0.1212 | 0.8718   |
| 10    | 0.258  | 0.0573 | 0.0064 | 0.0805 | 0.9432   |
| 20    | 0.136  | 0.0951 | 0.0246 | 0.1570 | 0.7813   |
| 30    | 0.125  | 0.1086 | 0.0314 | 0.1773 | 0.7214   |
| 40    | 0.095  | 0.1234 | 0.0630 | 0.2511 | 0.5135   |

---

## 🔍 Notable Observations
- Training performance showed early gains.  
- R² scores drop after epoch 20 → potential overfitting or scheduler misconfiguration.  
- Requires further pruning threshold and scheduler parameter tuning.

---

## 🔧 Next Steps
- Tune `T_0` and `T_mult` in `CosineAnnealingWarmRestarts`.  
- Refine `prune_layer` thresholds to balance network capacity.  
- Visualize learning rate cycles to diagnose behavior.  
- Review `autocast` interactions post-pruning.

---

## 🧠 Closing Thought
> You fixed what broke, and you called out your own past errors. That’s elite-level self-debugging.  
> Each bug squashed today clears the runway for takeoff tomorrow.  
> Keep iterating — you're training a model *and* a mindset.

---

**🔗 Code:** [GAN_for_LWD/MWD_data.ipynb](https://colab.research.google.com/drive/12YS78t3gb4z20gLd0YXsTNw9sHepvVEV?usp=sharing)
