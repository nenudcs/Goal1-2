# 模态槽位顺序固定为 T1CE, T2, FLAIR
MODALITIES = ["T1CE", "T2", "FLAIR"]

BINARY_FIELDS = {
    "Glioma": {"No": 0, "Yes": 1},
    "Enhancement": {"false": 0, "true": 1},
    "Necrosis": {"false": 0, "true": 1},
    "CysticChange": {"false": 0, "true": 1},
    "Hemorrhage": {"false": 0, "true": 1},
    "Calcification": {"false": 0, "true": 1},
    "Margin": {"Unclear": 0, "Clear": 1},
    "Lobulation": {"false": 0, "true": 1},
    "Morphology": {"Regular": 0, "Irregular": 1},
}

CATEGORICAL_FIELDS = {
    "WHO_grade": ["1", "2", "3", "4"],
    "EnhancementPattern": ["None", "Ring", "RimEnhancing", "Nodular", "GroundGlass", "Gyriform", "Multifocal", "Other"],
    "Signal_T2WI": ["Low", "Iso", "High"],
    "Signal_FLAIR": ["Low", "Iso", "High"],
}

LOCATION_CLASSES = [
    "RightFrontal", "RightTemporal", "RightCerebellar", "RightParietal", "RightOccipital", "RightBasalGanglia",
    "LeftFrontal", "LeftTemporal", "LeftCerebellar", "LeftParietal", "LeftOccipital", "LeftBasalGanglia",
    "Brainstem", "Other"
]
