from __future__ import annotations

from typing import Any, Dict

from .interfaces import ClassifierPlugin


class VetClinicClassifierPlugin(ClassifierPlugin):
    """Example classifier plug-in for the 'vet_clinic' domain.

    Very simple heuristic classifier for demonstration and testing.
    """

    name = "vet_clinic_classifier"

    def classify(self, item: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
        name = (item.get("name") or "").lower()
        address = (item.get("address") or "").lower()
        text = f"{name} {address}"
        if any(tok in text for tok in ("vet", "veterinary", "clinic")):
            return {"label": "vet_clinic", "score": 0.9}
        # Fallback to independent as default
        return {"label": "independent", "score": 0.6}
