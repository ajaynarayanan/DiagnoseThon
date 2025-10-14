SYSTEM_PROMPT = """
I am trying to win a contest on prompting LLMs with patient cases and getting likely diseases or tests (see contest info below). 
You are required to output a prompt for LLM, the required medical answer based on the prompt, and the exact disease name from the context provided to you (see SAMPLE OUTPUT for an example output). 
Make sure your prompts are detailed and try to target for medical professional track. 
You should not reveal the disease name in the prompt but give just enough details so that an LLM can come up with the correct disease. 

### CONTEST INFO ### 
To participate in the challenge, follow the four steps below and watch this short video for guidance on how to make your entry shine. Note that participants can submit as many diagnoses as they'd like to any of the tracks, which increases your chances of winning prizes (see the section below on prizes for more information).

1. Select Your LLM

Select one of these three LLMs to test your medical queries:

    ChatGPT 3.5
    Groq
    Gemini 1.5

You are free to choose other LLMs, in which you can select "Other" as the LLM type in the Diagnose-a-thon submission form.
2. Select Your Track

    Patient Track: Step into the shoes of a potential patient who might use a large language model (LLM) to get medical advice. In this track, use an LLM to get a diagnosis for real or imaginary symptoms.
    Medical Professional Track: Imagine you’re a doctor, nurse, or other medical professional who might use LLMs as a medical case assistant. In this track, use an LLM to receive a diagnosis based on a hypothetical patient case.
    Out-of-the-Box Track: Get creative! Think of a scenario not included in the Patient or Medical Professional Tracks that might lead to a potentially believable medical diagnosis from an LLM. For example, you may want to see whether an LLM can tell you about serious side effects of a medication that was recently prescribed, or you are interested in getting a second opinion on a diagnosis from an LLM.

3. Validate Your Findings

Find at least one trusted online source, such as academic or magazine articles, WebMD, or Wikipedia, to show that your AI-generated diagnosis is either accurate or misleading and harmful. You can submit both verifiably correct and incorrect diagnoses!
4. Submit Your Entries

Submit your entries using the submission form below. All entries should include:

    a screenshot of your LLM prompt
    the LLM-generated response or diagnosis
    the LLM tool and version used to generate your diagnosis
    a Word document that describes the online sources which validate or disprove the diagnosis you received (please include clickable links to those sources)
    if necessary, an explanation of why the diagnosis is particularly noteworthy, either in terms of being super-smart in diagnosing something or in terms of being dangerously misleading or potentially harmful to the user.

Once the competition concludes, participants will be asked to complete a five-minute survey about the experience. This survey is required of all participants who wish to be considered for the awards.

Participation in the Diagnose-a-thon is voluntary and can be ended at any time.


### SAMPLE OUTPUT ### 
### Prompt:

"I have a 29-year-old female patient, a kindergarten teacher, presenting with progressive neurological symptoms over the past few weeks. She initially complained of frequent, intense headaches and episodes of aphasia, specifically difficulty in finding words and forming sentences, which has begun to impact her daily work as a teacher. More recently, she has developed mild right-sided muscle weakness and occasional muscle twitches. Her symptoms are sporadic but worsening over time.

Initial workup included:

    CT Scan: Clear, no abnormalities noted.
    Basic Blood Work: All within normal limits, with no signs of infection or systemic illness.
    MRI (non-contrast): No evidence of brain tumor, stroke, or lesions typically associated with multiple sclerosis.

The patient has no significant medical history and is not on any regular medications. She has no known family history of neurological disease. Her physical exam shows mild aphasia and slightly diminished strength on her right side. Reflexes are normal, and no other focal neurological deficits are observed.

Given these findings, what possible differential diagnoses should be considered, and what further diagnostic tests would you recommend to narrow down the cause of her symptoms?"

### Expected LLM Response:

"In this case, the patient's progressive neurological symptoms—aphasia, muscle weakness, and headaches—could indicate an underlying neurological or infectious condition that might not be evident on initial imaging. Here are some differential diagnoses and recommended next steps:
Differential Diagnoses:

    Neurocysticercosis: This is a parasitic infection caused by Taenia solium larvae (pork tapeworm) and can lead to cyst formation in the brain. It’s prevalent in certain regions and can cause symptoms like seizures, focal neurological deficits, and cognitive changes. Neurocysticercosis may not always be visible on a non-contrast MRI, particularly if the cysts are small or calcified.

    Autoimmune Encephalitis: Conditions like anti-NMDA receptor encephalitis can present with progressive neurological symptoms, including aphasia and cognitive impairment. These conditions can sometimes be missed on initial imaging.

    Vascular Causes (such as Moyamoya Disease or Vasculitis): Chronic, progressive vascular conditions can lead to transient ischemic episodes and neurological symptoms that don’t show immediately on imaging.

    Migrainous Aphasia or Complex Migraine Variants: Certain types of migraines can cause episodic aphasia and neurological symptoms, though they are less likely to produce progressive weakness.

Recommended Diagnostic Tests:

    MRI with Contrast: A contrast-enhanced MRI can provide a clearer picture and may reveal infections, small vascular abnormalities, or lesions not visible on non-contrast imaging.

    Lumbar Puncture (LP): To assess for potential infections (including parasitic or autoimmune encephalitis) or other inflammatory markers, a cerebrospinal fluid (CSF) analysis could be helpful. If neurocysticercosis is suspected, the LP may also reveal elevated eosinophils or specific antibodies.

    Blood Tests: Consider specialized serologic tests for parasitic infections, autoimmune markers (such as ANA and anti-NMDA antibodies), and markers of inflammation or vasculitis.

    EEG (Electroencephalogram): This test can identify any seizure activity, which could correlate with her symptoms and may be indicative of neurocysticercosis or other encephalopathies.

Additional Considerations:

Review her travel history or possible exposure to regions where neurocysticercosis is more prevalent, as this could increase the likelihood of a parasitic infection. A review of dietary history might also help, particularly with any potential exposure to undercooked pork.

While this case requires further investigation, neurocysticercosis should remain high on the differential due to her neurological symptoms and the potential for missed detection in standard imaging."

### DISEASE:
Neurocysticercosis
"""

BASE_URL = "https://house.fandom.com/wiki/"


HOUSE_EPISODE_TITLES = [
    "Pilot",
    "Paternity",
    "Occam's Razor",
    "Maternity",
    "Damned If You Do",
    "The Socratic Method",
    "Fidelity",
    "Poison",
    "DNR",
    "Histories",
    "Detox",
    "Sports Medicine",
    "Cursed",
    "Control",
    "Mob Rules",
    "Heavy",
    "Role Model",
    "Babies & Bathwater",
    "Kids",
    "Love Hurts",
    "Three Stories",
    "Honeymoon",
    "Acceptance",
    "Autopsy",
    "Humpty Dumpty",
    "TB or Not TB",
    "Daddy's Boy",
    "Spin",
    "Hunting",
    "The Mistake",
    "Deception",
    "Failure to Communicate",
    "Need to Know",
    "Distractions",
    "Skin Deep",
    "Sex Kills",
    "Clueless",
    "Safe",
    "All In",
    "Sleeping Dogs Lie",
    "House vs. God",
    "Euphoria (Part 1)",
    "Euphoria (Part 2)",
    "Forever",
    "Who%27s_Your_Daddy%3F",
    "No Reason",
    "Meaning",
    "Cane & Able",
    "Informed Consent",
    "Lines in the Sand",
    "Fools for Love",
    "Que Sera Sera",
    "Son of Coma Guy",
    "Whac-A-Mole",
    "Finding Judas",
    "Merry Little Christmas",
    "Words and Deeds",
    "One Day, One Room",
    "Needle in a Haystack",
    "Insensitive",
    "Half-Wit",
    "Top Secret",
    "Fetal Position",
    "Airborne",
    "Act Your Age",
    "House Training",
    "Family",
    "Resignation",
    "The Jerk",
    "Human Error",
    "Alone",
    "The Right Stuff",
    "97_Seconds",
    "Guardian Angels",
    "Mirror Mirror",
    "Whatever It Takes",
    "Ugly",
    "You Don't Want to Know",
    "Games",
    "It's a Wonderful Lie",
    "Frozen",
    "Don't Ever Change",
    "No More Mr. Nice Guy",
    "Living the Dream",
    "House's Head",
    "Wilson's Heart",
    "Dying Changes Everything",
    "Not Cancer",
    "Adverse Events",
    "Birthmarks",
    "Lucky Thirteen",
    "Joy",
    "The Itch",
    "Emancipation",
    "Last Resort",
    "Let Them Eat Cake",
    "Joy to the World",
    "Painless",
    "Big Baby",
    "The Greater Good",
    "Unfaithful",
    "The Softer Side",
    "The Social Contract",
    "Here Kitty",
    "Locked In",
    "Simple Explanation",
    "Saviors",
    "House Divided",
    "Under My Skin",
    "Both Sides Now",
    "Broken",
    "Epic Fail",
    "The Tyrant",
    "Instant Karma",
    "Brave Heart",
    "Known Unknowns",
    "Teamwork",
    "Ignorance is Bliss",
    "Wilson",
    "The Down Low",
    "Remorse",
    "Moving the Chains",
    "5 to 9",
    "Private Lives",
    "Black Hole",
    "Lockdown",
    "Knight Fall",
    "Open and Shut",
    "The Choice",
    "Baggage",
    "Help Me",
    "Now_What%3F",
    "Selfish",
    "Unwritten",
    "Massage Therapy",
    "Unplanned Parenthood",
    "Office Politics",
    "A Pox on Our House",
    "Small Sacrifices",
    "Larger than Life",
    "Carrot or Stick",
    "Family Practice",
    "You Must Remember This",
    "Two Stories",
    "Recession Proof",
    "Bombshells",
    "Out of the Chute",
    "Fall From Grace",
    "The Dig",
    "Last Temptation",
    "Changes",
    "The Fix",
    "After Hours",
    "Moving On",
    "Twenty Vicodin",
    "Transplant",
    "Charity Case",
    "Risky Business",
    "The Confession",
    "Parents",
    "Dead & Buried",
    "Perils of Paranoia",
    "Better Half",
    "Runaways",
    "Nobody's Fault",
    "Chase",
    "Man of the House",
    "Love is Blind",
    "Blowing the Whistle",
    "Gut Check",
    "We Need the Eggs",
    "Body and Soul",
    "The_C_Word",
    "Post Mortem",
    "Holding On",
    "Everybody Dies",
]