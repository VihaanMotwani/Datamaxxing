import streamlit as st
import os
from Init_integrate import (
    download_instagram_reel,
    extract_frames_from_video,
    extract_text_from_frames,
    extract_audio_from_video,
    transcribe_audio_with_whisper
)
from integrated_system import IntegratedSystem

# Constants
VIDEO_FILE = "video.mp4"
AUDIO_FILE = "audio.mp3"
JSON_FILE = "data.json"

st.title("DeepContext")

# Input field for video URL
video_url = st.text_input("Enter Instagram Reel URL")

if st.button("Process Reel"):
    if video_url:
        status = st.empty()  # Create a placeholder for updating text
        
        status.write("Downloading video...")
        download_instagram_reel(video_url)

        if os.path.exists(VIDEO_FILE):
            status.success("Video downloaded successfully!")

            status.write("Extracting frames...")
            if extract_frames_from_video(VIDEO_FILE):
                status.success("Frames extracted successfully!")

                status.write("Extracting text from frames...")
                extract_text_from_frames()
                status.success("Text extraction complete!")

            status.write("Extracting audio and transcribing...")
            if extract_audio_from_video(VIDEO_FILE, AUDIO_FILE):
                transcribe_audio_with_whisper(AUDIO_FILE)
                status.success("Audio transcription complete!")

            status.success("Processing complete!")

            # Display extracted text & transcription
            if os.path.exists(JSON_FILE):
                
                # Run misinformation analysis
                status.write("Running misinformation analysis...")
                system = IntegratedSystem()
                result = system.analyze_json_file(JSON_FILE)

                st.subheader("Misinformation Analysis:")
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    contains_misinformation = result["misinformation_analysis"].get("contains_misinformation", False)
                    confidence_score = result["misinformation_analysis"].get("confidence_score", 0.0)
                    detected_criteria = result["misinformation_analysis"].get("detected_criteria", [])
                    explanation = result["misinformation_analysis"].get("explanation", "No explanation provided.")
                    web_context = result.get("web_context", {})

                    st.markdown(f"**Misinformation Detected:** {'✅ Yes' if contains_misinformation else '❌ No'}")
                    st.markdown(f"**Confidence Score:** {confidence_score * 100:.1f}%")
                    st.markdown(f"**Explanation:** {explanation}")

                    # Display Web Context in a formatted manner
                    if web_context and "error" not in web_context:
                        st.subheader("🌍 Web Context Analysis")
                        st.markdown(f"**Claim Analyzed:** {web_context.get('claim', 'No claim provided')}")

                        st.markdown("### **📌 Summary**")
                        st.write(web_context.get("context_summary", "No summary available."))

                        st.markdown("### **📜 Different Perspectives**")
                        for i, perspective in enumerate(web_context.get("perspectives", [])):
                            st.markdown(f"**Perspective {i+1}:** {perspective.get('viewpoint', 'No viewpoint provided')}")
                            st.markdown(f"- **Supporting Evidence:** {perspective.get('supporting_evidence', 'No evidence provided')}")
                            st.markdown(f"- **Limitations:** {perspective.get('limitations', 'No limitations mentioned')}")
                            st.write(" ")

                        st.markdown("### **🔬 Scientific Consensus**")
                        st.write(web_context.get("scientific_consensus", "No scientific consensus available."))

                        st.markdown("### **📖 Conclusion**")
                        st.write(web_context.get("conclusion", "No conclusion available."))

                        st.markdown("### **🔍 Information Gaps**")
                        st.write(web_context.get("information_gaps", "No missing information identified."))

                        st.markdown("### **🔗 Sources**")
                        for source in web_context.get("sources", []):
                            reliability = f" (Reliability Score: {source.get('reliability_score', 'Not evaluated')}/10)" if "reliability_score" in source else ""
                            st.markdown(f"- {source.get('title', 'Unknown')} {source.get('link', '#')} {reliability}")

                    else:
                        st.info("No additional web context found.")
