"""
Standalone Demo - FAST Stroke Detection Dashboard
Run with: streamlit run demo_dashboard.py
"""

import streamlit as st
import numpy as np
import time
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import tempfile
from pathlib import Path

# Try to import cv2, but don't fail if unavailable
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("⚠️ OpenCV not available. Using simulated analysis only.")

# Disable MediaPipe for cloud deployment
MEDIAPIPE_AVAILABLE = False

# Page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="FAST Stroke Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .alert-high {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(220, 53, 69, 0.3);
    }
    .alert-moderate {
        background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%);
        color: #000;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(255, 193, 7, 0.3);
    }
    .alert-low {
        background: linear-gradient(135deg, #28a745 0%, #218838 100%);
        color: white;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(40, 167, 69, 0.3);
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


class FaceAnalyzerDemo:
    """Demo face analyzer - simulated for cloud deployment"""
    
    def __init__(self):
        self.face_mesh = None
    
    def analyze_video(self, video_path: str, progress_callback=None) -> dict:
        """Simulated analysis for demo"""
        return self._simulated_analysis(progress_callback)
    
    def _simulated_analysis(self, progress_callback) -> dict:
        """Simulated analysis for demo"""
        # Simulate processing time
        for i in range(20):
            time.sleep(0.05)
            if progress_callback:
                progress_callback((i + 1) / 20)
        
        # Generate realistic demo data
        max_asymmetry = np.random.uniform(0.25, 0.55)
        asymmetry_timeline = list(np.random.uniform(0.15, max_asymmetry, 20))
        
        return {
            'mean_asymmetry': float(np.mean(asymmetry_timeline)),
            'max_asymmetry': float(max_asymmetry),
            'mean_smile_intensity': 0.08,
            'abnormal': max_asymmetry > 0.35,
            'confidence': 0.87,
            'frames_analyzed': 20,
            'asymmetry_timeline': asymmetry_timeline
        }


class StrokeDashboardDemo:
    """Demo version of stroke detection dashboard"""
    
    def __init__(self):
        self.initialize_session_state()
        self.face_analyzer = FaceAnalyzerDemo()
    
    def initialize_session_state(self):
        """Initialize session state"""
        defaults = {
            'current_step': 1,
            'face_result': None,
            'arms_result': None,
            'speech_result': None,
            'time_result': None,
            'final_prediction': None,
            'patient_id': f"DEMO-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render_header(self):
        """Render header"""
        st.markdown('<h1 class="main-header">🏥 FAST Stroke Detection System</h1>', 
                   unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-Powered Emergency Stroke Assessment - DEMO VERSION</p>', 
                   unsafe_allow_html=True)
        
        st.info("""
        💡 **Demo Mode:** This is a demonstration version showcasing the XAI interface and workflow.
        """)
        
        st.error("""
        🚨 **EMERGENCY MEDICAL TOOL** - If stroke is suspected, call emergency services (911/112) immediately.
        """)
    
    def render_progress_tracker(self):
        """Progress tracker"""
        steps = [
            ('😊 Face', 1),
            ('💪 Arms', 2),
            ('🗣️ Speech', 3),
            ('⏱️ Time', 4),
            ('📊 Results', 5)
        ]
        
        cols = st.columns(5)
        
        for col, (label, step_num) in zip(cols, steps):
            with col:
                if st.session_state.current_step > step_num:
                    st.markdown(f"""
                    <div style="text-align: center; color: #28a745;">
                        <div style="font-size: 2rem;">✅</div>
                        <div style="font-size: 0.9rem; font-weight: bold;">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif st.session_state.current_step == step_num:
                    st.markdown(f"""
                    <div style="text-align: center; color: #1f77b4;">
                        <div style="font-size: 2rem;">▶️</div>
                        <div style="font-size: 0.9rem; font-weight: bold;">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="text-align: center; color: #999;">
                        <div style="font-size: 2rem;">⭕</div>
                        <div style="font-size: 0.9rem;">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    def render_face_assessment(self):
        """Face assessment with demo data"""
        st.header("😊 Step 1: Face Drooping Detection")
        
        st.markdown("""
        ### Instructions:
        1. Click button below to generate simulated facial analysis
        2. In production, this would analyze a video of the patient smiling
        3. The AI detects facial asymmetry automatically
        """)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.info("🎭 **Demo Mode:** Click below to generate simulated facial analysis data")
            
            if st.button("🎲 Generate Demo Facial Analysis", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress):
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing... {int(progress * 100)}%")
                
                result = self.face_analyzer.analyze_video(None, update_progress)
                
                progress_bar.empty()
                status_text.empty()
                
                st.session_state.face_result = result
                st.success("✅ Analysis complete!")
                st.rerun()
        
        with col2:
            st.markdown("### 📊 Analysis Results")
            
            if st.session_state.face_result:
                result = st.session_state.face_result
                
                st.metric(
                    "Asymmetry Score",
                    f"{result['max_asymmetry']:.3f}",
                    delta="Abnormal" if result['abnormal'] else "Normal",
                    delta_color="inverse" if result['abnormal'] else "normal"
                )
                
                st.metric("Confidence", f"{result['confidence']:.1%}")
                st.metric("Frames Analyzed", f"{result['frames_analyzed']}")
                
                if result['abnormal']:
                    st.error("⚠️ **Facial asymmetry detected**")
                else:
                    st.success("✅ **Normal facial symmetry**")
                
                # Timeline
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=result['asymmetry_timeline'],
                    mode='lines',
                    name='Asymmetry',
                    line=dict(color='#dc3545', width=2)
                ))
                fig.add_hline(y=0.35, line_dash="dash", line_color="orange", annotation_text="Threshold")
                fig.update_layout(
                    title="Asymmetry Over Time",
                    xaxis_title="Frame",
                    yaxis_title="Score",
                    height=250,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("Continue to Arms →", type="primary", use_container_width=True):
                    st.session_state.current_step = 2
                    st.rerun()
            else:
                st.info("👆 Generate demo data above")
    
    def render_arms_assessment(self):
        """Arms assessment"""
        st.header("💪 Step 2: Arm Weakness Assessment")
        
        st.markdown("""
        ### Instructions:
        Ask patient to extend both arms forward with eyes closed for 10 seconds.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Left Arm")
            left_arm = st.radio(
                "Status:",
                ["Normal", "Mild drift", "Severe drift", "No movement"],
                key="left_arm"
            )
        
        with col2:
            st.subheader("Right Arm")
            right_arm = st.radio(
                "Status:",
                ["Normal", "Mild drift", "Severe drift", "No movement"],
                key="right_arm"
            )
        
        if st.button("✅ Record Arms Assessment", type="primary", use_container_width=True):
            scores = {"Normal": 0, "Mild drift": 1, "Severe drift": 2, "No movement": 3}
            abnormal = left_arm != "Normal" or right_arm != "Normal"
            
            st.session_state.arms_result = {
                'left_arm': left_arm,
                'right_arm': right_arm,
                'left_arm_score': scores[left_arm],
                'right_arm_score': scores[right_arm],
                'abnormal': abnormal,
                'severity': 'severe' if max(scores[left_arm], scores[right_arm]) >= 2 else 'mild' if abnormal else 'normal'
            }
            st.success("✅ Recorded!")
            time.sleep(0.5)
            st.session_state.current_step = 3
            st.rerun()
        
        if st.button("← Back"):
            st.session_state.current_step = 1
            st.rerun()
    
    def render_speech_assessment(self):
        """Speech assessment"""
        st.header("🗣️ Step 3: Speech Difficulty Assessment")
        
        st.markdown("""
        ### Test Phrase:
        Ask patient to repeat: *"You can't teach an old dog new tricks"*
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            articulation = st.radio(
                "Speech clarity:",
                ["Clear", "Mild slurring", "Moderate slurring", "Unintelligible"],
                key="articulation"
            )
        
        with col2:
            comprehension = st.radio(
                "Understanding:",
                ["Fully understands", "Partially", "Minimal", "No understanding"],
                key="comprehension"
            )
        
        if st.button("✅ Record Speech Assessment", type="primary", use_container_width=True):
            scores = {
                "Clear": 0, "Mild slurring": 1, "Moderate slurring": 2, "Unintelligible": 3,
                "Fully understands": 0, "Partially": 1, "Minimal": 2, "No understanding": 3
            }
            abnormal = articulation != "Clear" or comprehension != "Fully understands"
            
            st.session_state.speech_result = {
                'articulation': articulation,
                'comprehension': comprehension,
                'articulation_score': scores[articulation],
                'comprehension_score': scores[comprehension],
                'abnormal': abnormal,
                'severity': 'severe' if max(scores[articulation], scores[comprehension]) >= 2 else 'mild' if abnormal else 'normal'
            }
            st.success("✅ Recorded!")
            time.sleep(0.5)
            st.session_state.current_step = 4
            st.rerun()
        
        if st.button("← Back"):
            st.session_state.current_step = 2
            st.rerun()
    
    def render_time_tracking(self):
        """Time tracking"""
        st.header("⏱️ Step 4: Time - Symptom Onset")
        
        st.warning("**Time is brain!** Knowing symptom onset is critical for treatment.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            onset_known = st.radio(
                "Time known?",
                ["Yes - Exact", "Yes - Approximate", "Unknown"],
                key="onset"
            )
            
            time_hours = None
            if "Yes" in onset_known:
                time_hours = st.number_input(
                    "Hours since symptom onset:",
                    min_value=0.0,
                    max_value=72.0,
                    value=2.0,
                    step=0.5
                )
                
                if time_hours < 4.5:
                    st.success(f"✅ Within thrombolytic window ({4.5 - time_hours:.1f}h remaining)")
                elif time_hours < 24:
                    st.warning("⚠️ Within thrombectomy window")
                else:
                    st.error("🚨 Beyond acute treatment windows")
        
        with col2:
            st.markdown("### Windows")
            st.markdown("""
            **Thrombolytic:** 0-4.5h
            **Thrombectomy:** 0-24h
            """)
        
        if st.button("✅ Record Time Info", type="primary", use_container_width=True):
            st.session_state.time_result = {
                'onset_status': onset_known,
                'time_since_onset_hours': time_hours,
                'within_thrombolytic_window': time_hours and time_hours < 4.5,
                'within_thrombectomy_window': time_hours and time_hours < 24
            }
            st.success("✅ Recorded!")
            time.sleep(0.5)
            st.session_state.current_step = 5
            st.rerun()
        
        if st.button("← Back"):
            st.session_state.current_step = 3
            st.rerun()
    
    def render_results(self):
        """Results with XAI"""
        st.header("📊 Assessment Results & AI Analysis")
        
        if st.session_state.final_prediction is None:
            with st.spinner("🧠 Running AI analysis..."):
                time.sleep(1.5)
                st.session_state.final_prediction = self.generate_prediction()
        
        result = st.session_state.final_prediction
        pred = result['prediction']
        exp = result['explanation']
        
        # Risk banner
        risk = pred['risk_level']
        if risk == "HIGH":
            st.markdown(f"""
            <div class="alert-high">
                <h1 style="margin:0; font-size:2.5rem;">🚨 HIGH STROKE RISK</h1>
                <h2 style="margin:0.5rem 0;">Probability: {pred['stroke_probability']:.1%}</h2>
                <p style="font-size:1.2rem; margin:0.5rem 0 0 0;"><strong>IMMEDIATE EMERGENCY ACTION</strong></p>
            </div>
            """, unsafe_allow_html=True)
        elif risk == "MODERATE":
            st.markdown(f"""
            <div class="alert-moderate">
                <h1 style="margin:0; font-size:2.5rem;">⚠️ MODERATE RISK</h1>
                <h2 style="margin:0.5rem 0;">Probability: {pred['stroke_probability']:.1%}</h2>
                <p style="font-size:1.2rem; margin:0.5rem 0 0 0;"><strong>URGENT EVALUATION NEEDED</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-low">
                <h1 style="margin:0; font-size:2.5rem;">✅ LOW RISK</h1>
                <h2 style="margin:0.5rem 0;">Probability: {pred['stroke_probability']:.1%}</h2>
                <p style="font-size:1.2rem; margin:0.5rem 0 0 0;"><strong>Continue Monitoring</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Stroke Probability", f"{pred['stroke_probability']:.1%}")
        with col2:
            st.metric("Confidence", f"{pred['confidence']:.1%}")
        with col3:
            st.metric("Risk Level", f"{risk}")
        with col4:
            st.metric("Rules Triggered", len(exp['fired_rules']))
        
        st.markdown("---")
        
        # FAST Summary
        st.subheader("🔍 FAST Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            face_status = "🔴 ABNORMAL" if st.session_state.face_result['abnormal'] else "🟢 Normal"
            st.markdown(f"""
            <div class="metric-card">
                <h3>😊 Face</h3>
                <p style="font-size:1.5rem; margin:0.5rem 0;">{face_status}</p>
                <p style="margin:0;">Asymmetry: {st.session_state.face_result['max_asymmetry']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            arms_status = "🔴 ABNORMAL" if st.session_state.arms_result['abnormal'] else "🟢 Normal"
            st.markdown(f"""
            <div class="metric-card">
                <h3>💪 Arms</h3>
                <p style="font-size:1.5rem; margin:0.5rem 0;">{arms_status}</p>
                <p style="margin:0;">Severity: {st.session_state.arms_result['severity'].title()}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            speech_status = "🔴 ABNORMAL" if st.session_state.speech_result['abnormal'] else "🟢 Normal"
            st.markdown(f"""
            <div class="metric-card">
                <h3>🗣️ Speech</h3>
                <p style="font-size:1.5rem; margin:0.5rem 0;">{speech_status}</p>
                <p style="margin:0;">Severity: {st.session_state.speech_result['severity'].title()}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # XAI Tabs
        st.header("🔬 Explainable AI Analysis")
        
        tab1, tab2, tab3 = st.tabs(["🧩 Logic Rules", "📊 Feature Importance", "🔗 Reasoning Chain"])
        
        with tab1:
            st.subheader("Probabilistic Logic Rules (Prolog Knowledge Base)")
            
            for rule in exp['fired_rules']:
                with st.expander(f"**{rule['rule'].replace('_', ' ').title()}** - {rule['probability']:.0%}", expanded=True):
                    st.markdown(f"**Evidence:** {rule['evidence']}")
                    st.markdown(f"**Weight:** {rule['weight']:.0%}")
                    st.progress(rule['probability'])
            
            # Chart
            if exp['fired_rules']:
                rules_df = pd.DataFrame(exp['fired_rules'])
                fig = px.bar(
                    rules_df,
                    x='probability',
                    y='rule',
                    orientation='h',
                    title="Rule Activation Strengths",
                    color='weight',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Feature Importance")
            
            importance_df = pd.DataFrame([
                {'Feature': k, 'Importance': v}
                for k, v in exp['feature_importance'].items()
            ]).sort_values('Importance', ascending=False)
            
            fig = go.Figure(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h',
                marker=dict(
                    color=importance_df['Importance'],
                    colorscale='RdYlGn_r',
                    showscale=True
                ),
                text=importance_df['Importance'].apply(lambda x: f"{x:.1%}"),
                textposition='auto'
            ))
            fig.update_layout(
                title="FAST Component Contributions",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Reasoning Chain")
            for step in exp['reasoning_chain']:
                st.markdown(step)
            
            st.info("""
            💡 **Neuro-Symbolic Approach:**
            - Neural Network: Extracts features from video
            - Probabilistic Logic: Medical rules encode clinical knowledge
            - DeepProbLog: Combines both for interpretable predictions
            """)
        
        st.markdown("---")
        
        # Recommendations
        st.header("🏥 Clinical Recommendations")
        
        if pred['recommendation'] == "IMMEDIATE_EMERGENCY":
            st.error("""
            ### 🚨 IMMEDIATE ACTIONS
            1. **Call 911/112 NOW**
            2. **Note exact time** symptoms began
            3. **Do NOT** give food/drink
            4. Keep patient calm and still
            5. Prepare medication list
            """)
        elif pred['recommendation'] == "URGENT_EVALUATION":
            st.warning("""
            ### ⚠️ URGENT EVALUATION
            1. Transport to hospital within 1 hour
            2. Go to stroke center if possible
            3. Bring medication list
            4. Document symptom timeline
            """)
        else:
            st.info("""
            ### 📋 MONITORING
            1. Schedule doctor appointment
            2. Watch for symptom changes
            3. Review risk factors
            4. Document baseline assessment
            """)
        
        # Export
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            import json
            report = {
                'patient_id': st.session_state.patient_id,
                'timestamp': result['timestamp'],
                'prediction': pred,
                'fast_results': {
                    'face': st.session_state.face_result,
                    'arms': st.session_state.arms_result,
                    'speech': st.session_state.speech_result
                }
            }
            st.download_button(
                "💾 Download Report (JSON)",
                json.dumps(report, indent=2),
                f"stroke_report_{st.session_state.patient_id}.json",
                "application/json",
                use_container_width=True
            )
        
        with col2:
            summary = f"""STROKE ASSESSMENT SUMMARY
Patient: {st.session_state.patient_id}
Risk: {risk} ({pred['stroke_probability']:.1%})
Recommendation: {pred['recommendation']}

FAST Results:
- Face: {'ABNORMAL' if st.session_state.face_result['abnormal'] else 'Normal'}
- Arms: {'ABNORMAL' if st.session_state.arms_result['abnormal'] else 'Normal'}
- Speech: {'ABNORMAL' if st.session_state.speech_result['abnormal'] else 'Normal'}
"""
            st.download_button(
                "📄 Download Summary",
                summary,
                f"summary_{st.session_state.patient_id}.txt",
                use_container_width=True
            )
        
        with col3:
            if st.button("🔄 New Assessment", type="primary", use_container_width=True):
                for key in ['face_result', 'arms_result', 'speech_result', 'time_result', 'final_prediction']:
                    st.session_state[key] = None
                st.session_state.current_step = 1
                st.session_state.patient_id = f"DEMO-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                st.rerun()
    
    def generate_prediction(self):
        """Generate demo prediction"""
        face = st.session_state.face_result
        arms = st.session_state.arms_result
        speech = st.session_state.speech_result
        time_data = st.session_state.time_result
        
        # Calculate probability
        face_prob = 0.3 if face['abnormal'] else 0.05
        arms_prob = 0.35 * (max(arms['left_arm_score'], arms['right_arm_score']) / 3)
        speech_prob = 0.25 * (max(speech['articulation_score'], speech['comprehension_score']) / 3)
        
        stroke_prob = min(0.98, face_prob + arms_prob + speech_prob + 0.1)
        
        if stroke_prob >= 0.7:
            risk = "HIGH"
            rec = "IMMEDIATE_EMERGENCY"
        elif stroke_prob >= 0.4:
            risk = "MODERATE"
            rec = "URGENT_EVALUATION"
        else:
            risk = "LOW"
            rec = "MONITORING"
        
        prediction = {
            'stroke_probability': stroke_prob,
            'risk_level': risk,
            'confidence': 0.87,
            'recommendation': rec
        }
        
        # Generate rules
        rules = []
        if face['abnormal']:
            rules.append({
                'rule': 'facial_asymmetry_detected',
                'probability': 0.85,
                'evidence': f"Asymmetry {face['max_asymmetry']:.3f} > 0.35",
                'weight': 0.30
            })
        
        if arms['abnormal']:
            rules.append({
                'rule': 'unilateral_arm_weakness',
                'probability': 0.82,
                'evidence': f"Arm weakness detected",
                'weight': 0.35
            })
        
        if speech['abnormal']:
            rules.append({
                'rule': 'speech_impairment',
                'probability': 0.78,
                'evidence': f"Speech difficulty detected",
                'weight': 0.25
            })
        
        if time_data.get('within_thrombolytic_window'):
            rules.append({
                'rule': 'time_critical_intervention',
                'probability': 0.95,
                'evidence': "Within 4.5h window",
                'weight': 0.10
            })
        
        reasoning = [
            f"1. FAST Assessment: Face={'ABN' if face['abnormal'] else 'Norm'}, Arms={'ABN' if arms['abnormal'] else 'Norm'}, Speech={'ABN' if speech['abnormal'] else 'Norm'}",
            f"2. {len(rules)} clinical rules triggered",
            f"3. Combined probability: {stroke_prob:.1%}",
            f"4. Risk: {risk}"
        ]
        
        importance = {
            'Face Asymmetry': face_prob / stroke_prob if stroke_prob > 0 else 0,
            'Arm Weakness': arms_prob / stroke_prob if stroke_prob > 0 else 0,
            'Speech Difficulty': speech_prob / stroke_prob if stroke_prob > 0 else 0,
            'Time Factor': 0.1 / stroke_prob if stroke_prob > 0 else 0
        }
        
        explanation = {
            'fired_rules': rules,
            'reasoning_chain': reasoning,
            'feature_importance': importance
        }
        
        return {
            'prediction': prediction,
            'explanation': explanation,
            'timestamp': datetime.now().isoformat()
        }
    
    def run(self):
        """Main entry"""
        self.render_header()
        self.render_progress_tracker()
        
        if st.session_state.current_step == 1:
            self.render_face_assessment()
        elif st.session_state.current_step == 2:
            self.render_arms_assessment()
        elif st.session_state.current_step == 3:
            self.render_speech_assessment()
        elif st.session_state.current_step == 4:
            self.render_time_tracking()
        elif st.session_state.current_step == 5:
            self.render_results()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem 0;">
            <p><strong>🎓 MSc Applied AI - Group Project Demo</strong></p>
            <p>Team: Elizabeth | Grace | Siska | Chito | Mais</p>
            <p style="color: #dc3545;">⚠️ Demo only - not for clinical use</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    app = StrokeDashboardDemo()
    app.run()