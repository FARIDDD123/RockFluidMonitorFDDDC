import pytest
import pandas as pd
import numpy as np
from data_generator import fluid_loss_risk, emulsion_risk, reactivity_score 

# نمونه‌ی کوچک از داده‌ها برای تست
@pytest.fixture
def test_row():
    return {
        "Viscosity_cP": 100,
        "Solid_Content_%": 10,
        "Oil_Water_Ratio": 70,
        "Emulsion_Stability": 80,
        "Shale_Reactiveness": "Medium"
    }

def test_fluid_loss_risk(test_row):
    val = fluid_loss_risk(test_row)
    expected = min(1, (100/120) * (10/20))
    assert np.isclose(val, expected)

def test_emulsion_risk(test_row):
    val = emulsion_risk(test_row)
    expected = (100 - 70)/100 + (100 - 80)/100
    assert np.isclose(val, expected)

def test_reactivity_score_high():
    assert reactivity_score({"Shale_Reactiveness": "High"}) == 1

def test_reactivity_score_medium():
    assert reactivity_score({"Shale_Reactiveness": "Medium"}) == 0.5

def test_reactivity_score_low():
    assert reactivity_score({"Shale_Reactiveness": "Low"}) == 0

# تست افزودن نویز
def test_add_noise_consistency():
    df = pd.DataFrame({
        'A': np.ones(1000) * 50,
        'B': np.ones(1000) * 100
    })
    from your_module_name import add_noise
    noisy_df = add_noise(df.copy(), ['A', 'B'], noise_level=0.1)
    assert not noisy_df.equals(df)  # تغییر باید رخ داده باشد
    assert (noisy_df['A'] != df['A']).any()
    assert (noisy_df['B'] != df['B']).any()

# تست افزودن داده‌های گمشده
def test_add_missing_data():
    df = pd.DataFrame({
        'X': np.random.rand(1000),
        'Y': np.random.randint(0, 100, 1000)
    })
    from your_module_name import add_missing_data
    df_with_nan = add_missing_data(df.copy(), missing_rate=0.05)
    nan_counts = df_with_nan.isna().sum()
    assert nan_counts['X'] > 0
    assert nan_counts['Y'] > 0
