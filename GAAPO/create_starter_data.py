"""Write synthetic, unreviewed storyboard fixtures; never overwrite existing data."""
import json
from pathlib import Path


def scene(number, narration, visual, text):
    return {'scene_number': number, 'narration': narration,
            'visual_description': visual, 'on_screen_text': text}


# Each split has different scenarios. These are authored examples, not real patient data.
SPECS = {
    'train': [
        ('Getting started with activity',
         'Create a three-scene patient video about discussing a new activity plan with the heart failure team before starting exercise.',
         [('An activity plan should fit your heart failure symptoms and your situation.', 'Patient and clinician discussing a calendar.', 'A plan that fits you'),
          ('Before starting a new exercise routine, discuss it with your heart failure team.', 'Patient bringing an activity question list to an appointment.', 'Ask before starting'),
          ('Ask whether cardiac rehabilitation would help you get started with support.', 'Clinician pointing to a rehabilitation programme leaflet.', 'Ask about cardiac rehabilitation')]),
        ('Understanding ICD activity settings',
         'Create a three-scene educational storyboard explaining why someone with an ICD should discuss device settings with their exercise professional.',
         [('An implantable cardioverter defibrillator, or ICD, treats certain dangerous heart rhythms.', 'Simple nontechnical drawing of a person with an implanted device.', 'Your ICD'),
          ('Ask your device team and exercise professional how your device settings affect your activity plan.', 'Device clinician and exercise professional discussing a patient-specific plan.', 'Know your personal plan'),
          ('Use the advice given for your device rather than a heart-rate limit from another person.', 'Two different personalised activity cards with no numeric limits.', 'Advice is individual')]),
        ('Planning around tiredness',
         'Create a three-scene video for a heart failure patient about planning activity around fatigue without giving a fixed exercise dose.',
         [('Tiredness can make activity feel difficult when you have heart failure.', 'Patient taking a comfortable seated break.', 'Talk about tiredness'),
          ('Discuss manageable activity goals with your care team and build your routine gradually as advised.', 'Patient and care team agreeing on a small goal.', 'Choose manageable goals'),
          ('If activity leaves you very fatigued, stop and discuss the symptoms with your clinician.', 'Patient stopping an activity and making a call.', 'Report troublesome symptoms')]),
        ('Talking about intimacy',
         'Create a three-scene sensitive video about talking with a partner and the care team about intimacy when living with heart failure.',
         [('It is understandable to have questions about intimacy when living with heart failure.', 'Couple having a private conversation in a calm room.', 'Questions are welcome'),
          ('Tell your partner how you feel and ask your care team about your individual concerns.', 'Couple preparing questions for a confidential consultation.', 'Talk openly'),
          ('Choose a time when you feel more comfortable and have more energy, and seek individual advice about symptoms.', 'Simple clock beside a private consultation card.', 'Comfort and individual advice')]),
        ('When activity symptoms change',
         'Create a three-scene patient storyboard explaining that severe breathlessness during exercise should not be ignored.',
         [('Notice how you feel during activity.', 'Patient checking in with their comfort during a walk.', 'Notice your symptoms'),
          ('Stop exercising if you become severely short of breath and discuss your symptoms with your doctor.', 'Patient stopping a walk and contacting the care team.', 'Stop and seek advice'),
          ('If symptoms are severe or you think it is an emergency, contact local emergency services.', 'Phone showing an emergency-call symbol without a country-specific number.', 'Emergency symptoms need urgent help')]),
        ('Questions for an activity appointment',
         'Create a three-scene storyboard for patients with heart failure and an ICD who want to prepare questions about physical activity, sport and sexual activity.',
         [('Bring your questions about physical activity, sport and sexual activity to your care team.', 'Patient writing three topic headings on a notebook.', 'Activity • sport • intimacy'),
          ('Ask how your symptoms, recent treatment and ICD affect your personal plan.', 'Patient reviewing their individual plan with the device team.', 'Your plan is individual'),
          ('Write down the advice you receive and who to contact if questions or symptoms arise.', 'Patient leaving with a written plan and contact card.', 'Keep your plan and contacts')])
    ],
    'validation': [
        ('A supported first step',
         'Make a two-scene reassuring video for someone with heart failure who is anxious about attending their first cardiac rehabilitation session.',
         [('Ask your care team what to expect from cardiac rehabilitation and whether it is suitable for you.', 'Patient discussing a first rehabilitation visit with a clinician.', 'Ask what to expect'),
          ('Share your worries and symptoms with the rehabilitation team so your support can be personalised.', 'Patient being welcomed and discussing concerns with rehabilitation staff.', 'Share your concerns')]),
        ('Returning to a favourite sport',
         'Make a two-scene storyboard about a patient with an ICD asking their device team about returning to a favourite sport; do not provide universal clearance.',
         [('Before returning to your sport, discuss the activity and your ICD with your device team.', 'Patient showing their sports equipment to a device clinician.', 'Ask your device team'),
          ('Ask for a personal plan that takes your condition and device settings into account.', 'Clinician and patient reviewing a written sports plan with no fixed thresholds.', 'Personal advice before sport')]),
        ('A private question about sex',
         'Make a two-scene video showing an older adult with heart failure asking a clinician about sexual activity without embarrassment or blanket reassurance.',
         [('You can ask your clinician about sexual activity in a private conversation.', 'Older adult speaking privately with a clinician.', 'You can ask privately'),
          ('Explain your concerns and any symptoms so the advice can fit your own situation.', 'Patient and clinician discussing a confidential question card.', 'Individual advice matters')]),
        ('Recording how a walk felt',
         'Make a two-scene educational video showing a heart failure patient recording activity questions and symptoms to discuss at a follow-up appointment.',
         [('Write down how your activity felt and any questions you want to raise.', 'Patient making a brief note after their usual activity.', 'Record your questions'),
          ('Bring these notes to your follow-up so you and the care team can discuss your activity plan.', 'Patient sharing a notebook during an appointment.', 'Discuss your plan')])
    ],
    'test': [
        ('A caregiver can help with questions',
         'Create a two-scene video for a caregiver helping a relative with heart failure and an ICD prepare for an activity consultation.',
         [('With your relative’s agreement, help them write down their questions about activity and their ICD.', 'Relative and caregiver writing a question list together.', 'Support their questions'),
          ('Encourage them to discuss their own symptoms and goals with the care team.', 'Relative leading a conversation with a clinician while the caregiver listens.', 'Their goals, their care plan')]),
        ('Choosing a realistic goal',
         'Create a two-scene plain-language storyboard for someone with heart failure who feels discouraged by unrealistic exercise goals.',
         [('Choose a manageable goal with your care team instead of comparing your progress with someone else’s.', 'Patient and clinician choosing a small personal goal on a blank planning card.', 'A manageable goal'),
          ('Review how you feel and ask the team how to adjust your plan gradually.', 'Patient returning to discuss progress and questions.', 'Review and adjust together')]),
        ('An intimacy conversation for two',
         'Create a two-scene non-explicit video for a couple coping with tiredness and worries about intimacy after a heart failure diagnosis.',
         [('Tell each other how you feel and what would help you feel comfortable.', 'Couple talking quietly in a private living room.', 'Talk and listen'),
          ('Discuss health concerns with your care team rather than guessing what is safe for you.', 'Couple preparing a question for a confidential consultation.', 'Ask for personal advice')]),
        ('Making an exercise contact card',
         'Create a two-scene storyboard for a heart failure patient making a contact card with their care team before following an activity plan.',
         [('Ask your care team who to contact about changes in symptoms during your activity plan.', 'Patient and clinician filling in a contact card together.', 'Know who to contact'),
          ('Keep the contact card with your written plan. For an emergency, use local emergency services.', 'Contact card beside an activity plan and a separate emergency-call icon.', 'Keep your contacts handy')])
    ]
}


def main():
    directory = Path(__file__).parent / 'data'
    directory.mkdir(exist_ok=True)
    paths = [directory / f'{split}.json' for split in SPECS]
    if any(path.exists() for path in paths):
        raise FileExistsError('Starter datasets already exist; existing data will not be overwritten')
    for split, specs in SPECS.items():
        rows = [{'id': f'synthetic-{split}-{i:03d}', 'user_request': request,
                 'output': {'title': title, 'scenes': [scene(n, *content)
                            for n, content in enumerate(scenes, 1)]}}
                for i, (title, request, scenes) in enumerate(specs, 1)]
        (directory / f'{split}.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'{split}: {len(rows)} examples written')
    provenance = {
        'status': 'synthetic_unreviewed_starter_data', 'clinician_reviewed': False,
        'purpose': 'Pipeline setup and development, not clinical validation or thesis benchmark results',
        'created': '2026-10-05', 'split_counts': {s: len(rows) for s, rows in SPECS.items()},
        'limitations': ['Small authored dataset', 'Related topics across splits; no claim of independent clinical coverage',
                       'Reference storyboards are provisional, not gold-standard clinical labels'],
        'sources': [
            'https://www.bhf.org.uk/informationsupport/heart-matters-magazine/activity/exercise-for-heart-failure',
            'https://www.bhf.org.uk/informationsupport/heart-matters-magazine/activity/exercising-with-a-heart-condition',
            'https://www.bhf.org.uk/informationsupport/support/practical-support/living-with-heart-failure',
            'https://hfsa.org/hfsa-patient-education-exercise-and-activity-heart-failure'
        ]}
    (directory / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
