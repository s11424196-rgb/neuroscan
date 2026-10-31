"""
About page for NeuroScan AI.

Provides detailed information about the models, supported formats,
how the AI works, known limitations, references, ethics statement,
and a contact placeholder.
"""

from __future__ import annotations

import streamlit as st

from src.frontend.components.disclaimer import show_disclaimer_banner


def render() -> None:
    """Render the about page."""
    show_disclaimer_banner()

    st.title("ℹ️ About NeuroScan AI")

    # ── Model Information ───────────────────────────────────────────────
    st.header("🧠 Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            #### Brain Tumour Detection
            | Property | Value |
            |---|---|
            | Architecture | EfficientNet-B0 |
            | Input Size | 224 × 224 px |
            | Classes | Glioma, Meningioma, Pituitary, No Tumour |
            | Training Images | ~7,023 |
            | Test Accuracy | ~96.2% |
            | Framework | PyTorch |
            """
        )

    with col2:
        st.markdown(
            """
            #### Alzheimer's Screening
            | Property | Value |
            |---|---|
            | Architecture | ResNet-50 |
            | Input Size | 224 × 224 px |
            | Classes | Non-Demented, Very Mild, Mild, Moderate |
            | Training Images | ~6,400 |
            | Test Accuracy | ~93.8% |
            | Framework | PyTorch |
            """
        )

    st.divider()

    # ── Supported Formats ───────────────────────────────────────────────
    st.header("📁 Supported Scan Formats")
    st.markdown(
        """
        | Format | Extension(s) | Notes |
        |---|---|---|
        | JPEG | `.jpg`, `.jpeg` | Common for pre-processed slices |
        | PNG | `.png` | Lossless; good for research exports |
        | DICOM | `.dcm` | Standard clinical MRI format |
        | NIfTI | `.nii`, `.nii.gz` | Neuroimaging research standard |

        > **Tip:** For best results, upload axial T1-weighted or
        > T2-weighted MRI slices at 224 × 224 pixels or higher resolution.
        > The system will resize images automatically.
        """
    )

    st.divider()

    # ── How the AI Works ────────────────────────────────────────────────
    st.header("⚙️ How It Works")
    st.markdown(
        """
        #### Convolutional Neural Networks (CNNs)

        NeuroScan AI uses **deep convolutional neural networks** — a class
        of machine-learning models that learn to recognise visual patterns
        directly from pixel data.

        1. **Input** — Your MRI scan is resized to 224 × 224 pixels and
           normalised.
        2. **Feature Extraction** — The CNN passes the image through dozens
           of convolutional layers that detect increasingly complex
           features: edges → textures → shapes → anatomical structures.
        3. **Classification** — The extracted features are fed into a
           fully-connected layer that outputs a probability for each class.
        4. **Prediction** — The class with the highest probability is
           reported along with its confidence score.

        #### Grad-CAM Explainability

        To help you understand *why* the model made a particular prediction,
        we use **Gradient-weighted Class Activation Mapping (Grad-CAM)**.

        Grad-CAM highlights the regions of the input image that most
        influenced the model's decision by computing gradients of the
        predicted class with respect to the final convolutional layer.
        The resulting heatmap is overlaid on the original scan so you can
        see which areas the model "looked at."

        > ⚠️ Grad-CAM is an *approximation* of model attention. It does
        > not guarantee that highlighted regions correspond to clinically
        > meaningful features.
        """
    )

    st.divider()

    # ── Known Limitations ───────────────────────────────────────────────
    st.header("⚠️ Known Limitations & Failure Cases")
    st.markdown(
        """
        - **Dataset Bias:** Models were trained on specific public datasets
          and may not generalise to scans from different scanners,
          protocols, or patient populations.
        - **Slice Selection:** Performance depends on the MRI slice chosen.
          Non-representative slices may lead to incorrect predictions.
        - **Resolution Sensitivity:** Very low-resolution or heavily
          compressed images degrade accuracy.
        - **Out-of-Distribution Inputs:** Non-brain images, CT scans, or
          corrupted files may produce meaningless but confident-looking
          predictions.
        - **Class Imbalance:** Some classes (e.g., Moderate Dementia) had
          fewer training samples, which may reduce recall for those classes.
        - **No Longitudinal Analysis:** The tool analyses individual scans
          in isolation and cannot track progression over time.
        - **No Clinical Context:** The AI has no access to patient history,
          symptoms, or other diagnostic information.
        """
    )

    st.divider()

    # ── References ──────────────────────────────────────────────────────
    st.header("📚 References & Datasets")
    st.markdown(
        """
        1. **Brain Tumor MRI Dataset** — Bhuvaji S., Kadam A., Bhumkar P.,
           Dedge S., Kanchan S. (2020). Available on
           [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
        2. **Alzheimer's MRI Dataset** — Dubey S. (2020). Available on
           [Kaggle](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images).
        3. **EfficientNet** — Tan M. & Le Q. V. (2019). *EfficientNet:
           Rethinking Model Scaling for Convolutional Neural Networks.*
           ICML 2019.
        4. **ResNet** — He K., Zhang X., Ren S., Sun J. (2015). *Deep
           Residual Learning for Image Recognition.* CVPR 2016.
        5. **Grad-CAM** — Selvaraju R. R. et al. (2017). *Grad-CAM: Visual
           Explanations from Deep Networks via Gradient-based
           Localization.* ICCV 2017.
        """
    )

    st.divider()

    # ── Ethics Statement ────────────────────────────────────────────────
    st.header("🤝 Ethics & Responsible AI")
    st.markdown(
        """
        We are committed to the ethical development and deployment of AI
        in healthcare research:

        - **Transparency:** We provide Grad-CAM explanations and publish
          model architectures and training details.
        - **No Clinical Claims:** We explicitly state that this tool is
          not validated for clinical use and must not replace professional
          diagnosis.
        - **Privacy:** Uploaded scans are processed in-session and are not
          stored permanently. In demo mode, all processing occurs locally.
        - **Fairness:** We acknowledge potential biases in training data
          and encourage validation across diverse populations before any
          clinical consideration.
        - **Open Science:** This project is intended to support open
          research and education in medical AI.
        """
    )

    st.divider()

    # ── Contact ─────────────────────────────────────────────────────────
    st.header("📬 Contact & Feedback")
    st.markdown(
        """
        This project is under active development. We welcome feedback,
        bug reports, and collaboration inquiries.

        - **GitHub:** [github.com/neuroscan-ai](https://github.com) *(placeholder)*
        - **Email:** neuroscan-ai@example.com *(placeholder)*
        - **Issues:** Please use the GitHub issue tracker for bug reports.
        """
    )

    st.divider()

    # ── Full Disclaimer ─────────────────────────────────────────────────
    st.header("📋 Full Disclaimer")
    st.markdown(
        """
        > **RESEARCH & EDUCATIONAL USE ONLY**
        >
        > NeuroScan AI is an experimental artificial intelligence tool
        > developed for research and educational purposes. The predictions
        > and visualisations it generates are produced by machine-learning
        > models that **may be inaccurate, biased, or incomplete**.
        >
        > **This tool does NOT constitute medical advice.** It has NOT been
        > validated for clinical use and must NOT be used as the sole basis
        > for any medical decision, diagnosis, or treatment plan.
        >
        > Any clinical interpretation of brain MRI scans should be
        > performed by a qualified radiologist or neurologist in the
        > context of a comprehensive clinical evaluation.
        >
        > The developers and operators of NeuroScan AI accept **no
        > liability** for any actions taken based on the output of this
        > tool.
        >
        > If you or someone you know has concerns about a neurological
        > condition, **please consult a licensed healthcare professional
        > immediately**.
        """
    )
