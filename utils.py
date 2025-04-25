request_response = {
    "Input": ["QUESTION"],  # "Objective", "Background Context"],
    "Output": [
        "Objective",
        "Background Context",
        "Methodology",
        # "Participants and Setting",
        # "Data",
    ],
}

cluster2label = {
    "Objective": [
        "PURPOSE",
        "OBJECTIVE",
        "OBJECTIVES",
        "AIMS",
        "AIMS AND OBJECTIVES",
        "PRIMARY OBJECTIVE",
        "PURPOSE OF INVESTIGATION",
        "STUDY OBJECTIVE",
        "BACKGROUND AND OBJECTIVES",
        "RATIONALE AND OBJECTIVES",
        "BACKGROUND AND OBJECTIVE",
        "RATIONALE",
        "RATIONALE, AIMS AND OBJECTIVES",
        "GOAL",
        "PURPOSE OF THE STUDY",
        "BACKGROUND AND STUDY OBJECTIVE",
        "BACKGROUND AND STUDY AIMS",
        "BACKGROUND AND AIM",
        "BACKGROUND AND AIM OF THE STUDY",
        "INTRODUCTION AND HYPOTHESIS",
        "CONTEXT AND OBJECTIVES",
        "BACKGROUND AND PURPOSE",
        "BACKGROUND AND AIMS",
        "AIMS AND BACKGROUND",
        "HYPOTHESIS",
        "HYPOTHESES",
        "OBJECT",
        "PURPOSES",
    ],
    "Background Context": [
        "BACKGROUND",
        "SUMMARY BACKGROUND DATA",
        "SUMMARY OF BACKGROUND DATA",
        "BACKGROUND CONTEXT",
        "INTRODUCTION",
        "CONTEXT",
        "SUMMARY",
    ],
    "Methodology": [
        "OBJECTIVE AND METHODS",
        "METHODS",
        "METHOD",
        "METHOD AND MATERIAL",
        "METHODS AND MATERIALS",
        "METHODS AND RESULTS",
        "METHODS AND STUDY DESIGN",
        "METHODOLOGY",
        "RESEARCH DESIGN",
        "RESEARCH DESIGN AND METHODS",
        "EXPERIMENTAL DESIGN",
        "DESIGN",
        "DESIGN OF STUDY",
        "STUDY DESIGN",
        "STUDY DESIGN AND METHODS",
        "STUDY DESIGN AND SETTING",
        "DESIGN, SETTING AND PATIENTS",
        "DESIGN, SETTING, AND PARTICIPANTS",
        "DESIGN AND SETTING",
        "PATIENTS AND METHOD",
        "MATERIAL",
        "MATERIALS AND METHODS",
        "MATERIAL AND METHODS",
        "MATERIAL AND METHOD",
        "STUDY POPULATION AND METHODS",
        "STUDY OBJECTIVES",
        "DESIGN, SETTING, AND PATIENTS",
    ],
    "Participants and Setting": [
        "PATIENTS",
        "PATIENT SAMPLE",
        "PATIENTS AND METHODS",
        "PATIENTS AND SETTING",
        "PATIENTS AND PARTICIPANTS",
        "PARTICIPANTS",
        "POPULATION",
        "POPULATION AND SAMPLE",
        "STUDY SAMPLE",
        "SUBJECTS",
        "PARTICIPANTS AND INTERVENTION",
        "SETTING AND PARTICIPANTS",
        "SETTINGS AND PARTICIPANTS",
        "PATIENT AND METHODS",
        "PROBANDS AND METHODS",
        "SUBJECTS AND METHODS",
        "LOCATION, SUBJECTS, AND INTERVENTIONS",
        "SETTING",
        "SETTINGS",
        "STUDY SETTING",
        "LOCATION",
        "SETTING AND PATIENTS",
        "SAMPLE",
        "STUDY SELECTION",
    ],
    "Data": [
        "DATA EXTRACTION",
        "DATA SOURCES",
        "DATA SOURCE",
        "DATA SYNTHESIS",
        "COLLECTION",
        "EXTRACTION METHODS",
    ],
    "Intervention and Treatment": [
        "INTERVENTION",
        "INTERVENTIONS",
        "INTERVENTIONS AND OUTCOME MEASURES",
        "METHODS OR INTERVENTIONS",
    ],
    "Results": [
        "MAIN OUTCOME",
        "MAIN OUTCOME MEASURE",
        "MAIN OUTCOME MEASURES",
        "MAIN OUTCOME MEASUREMENTS",
        "OUTCOME",
        "OUTCOME MEASURE",
        "OUTCOME MEASURES",
        "OUTCOMES",
        "OUTCOMES MEASURED",
        "RESULT",
        "RESULTS",
        "MAIN RESULTS",
        "KEY RESULTS",
        "RESULTS AND DISCUSSION",
        "RESULTS AND LIMITATIONS",
        "MEASUREMENTS",
        "MEASURES",
        "MEASUREMENTS AND RESULTS",
        "MEASUREMENT AND RESULTS",
        "MEASUREMENTS AND MAIN RESULTS",
        "MAIN MEASURES",
        "MAIN RESEARCH VARIABLES",
        "FINDINGS",
        "OBSERVATIONS",
        "PRIMARY AND SECONDARY OUTCOME MEASURES",
        "PRINCIPAL FINDINGS",
        "DISCUSSION",
        "END POINT",
    ],
    "Case Reports and Descriptions": [
        "CASE REPORT",
        "CASE REPORTS",
        "CASE PRESENTATION",
        "CASE DESCRIPTION",
        "CASE DEFINITION",
        "DESCRIPTION",
        "PRESENTATION",
        "MAIN BODY",
    ],
    "Trial and Registration": ["TRIAL REGISTRATION", "CLINICAL TRIAL REGISTRATION"],
    "Limitations": ["LIMITATIONS", "KEY LIMITATIONS"],
    "Miscellaneous": [
        "ICU LOS",
        ", FSAII",
        ", SCCVII",
        "UNLABELLED",
    ],
}


def get_prompt(question):
    user_input = f"""Given the following research question, please provide a methodology to conduct research on it which would enable the researcher to answer the question. 

Research Question: {question}
    
Please be concise within 2-4 sentences for Introduction and Methodology. Do not use markdown format. Format the output as:
Introduction:
Methodology:
"""
    return user_input


def get_output_format(introduction, methodology):
    output_format = f"""Introduction: {introduction}

Methodology: {methodology}

"""
    return output_format


def get_gemma_messages_prompt_resp(question, introduction, methodology):
    user_input = get_prompt(question)
    messages = [
        [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input},
                ],
            },
            {
                "role": "model",
                "content": [
                    {
                        "type": "text",
                        "text": get_output_format(introduction, methodology),
                    },
                ],
            },
        ],
    ]

    return messages


def get_gemma_messages(question):
    user_input = get_prompt(question)
    messages = [
        [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input},
                ],
            },
        ],
    ]

    return messages


def get_smollmv2_messages(question):
    user_input = get_prompt(question)
    messages = [{"role": "user", "content": user_input}]
    return messages


def get_llama_messages_prompt_resp(question, introduction, methodology):
    user_input = get_prompt(question)
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_input},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": get_output_format(introduction, methodology),
                },
            ],
        },
    ]

    return messages


def extract_gemma_response(decoded_text, model="model"):
    parts = decoded_text.split(f"<start_of_turn>{model}\n")
    if len(parts) > 1:
        return parts[-1].split("<end_of_turn>")[0].strip()
    return decoded_text


def extract_smollmv2_response(decoded_text, model="assistant"):
    parts = decoded_text.split(f"<|im_start|>{model}\n")
    if len(parts) > 1:
        return parts[-1].split("<|im_end|>")[0].strip()
    return decoded_text


def extract_llama_response(decoded_text):
    parts = decoded_text.split(f"<|start_header_id|>assistant<|end_header_id|>\n")
    if len(parts) > 1:
        return parts[-1].split("<|eot_id|>")[0].strip()
    return decoded_text


get_messages = {
    "google/gemma-3-1b-it": get_gemma_messages,
    "HuggingFaceTB/SmolLM2-135M-Instruct": get_smollmv2_messages,
    "meta-llama/Llama-3.2-1B-Instruct": get_smollmv2_messages,
}

extract_assistant_response = {
    "google/gemma-3-1b-it": extract_gemma_response,
    "HuggingFaceTB/SmolLM2-135M-Instruct": extract_smollmv2_response,
    "meta-llama/Llama-3.2-1B-Instruct": extract_llama_response,
}

get_messages_prompt_resp = {
    "google/gemma-3-1b-it": get_gemma_messages_prompt_resp,
    # "HuggingFaceTB/SmolLM2-135M-Instruct": get_smollmv2_messages,
    "meta-llama/Llama-3.2-1B-Instruct": get_llama_messages_prompt_resp,
}
