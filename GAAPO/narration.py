"""Prepare spoken narration without changing visual captions or raw model outputs."""
import copy
import re

# Explicit initialisms only: do not spell out pronounceable words such as NYHA.
INITIALISMS = ('ICD', 'S-ICD', 'CRT', 'CRT-D', 'CRT-P', 'ECG', 'EKG', 'ATP',
               'VT', 'VF', 'LVEF', 'EF', 'ARB', 'MRA', 'SGLT2',
               'CPR', 'AED', 'MRI', 'CT', 'AV', 'SA', 'AF', 'HF')
PATTERN = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(x) for x in
                     sorted(INITIALISMS, key=len, reverse=True)) + r')(?!\w)')

NARRATION_RULE = ('For spoken narration, write medical initialisms as separate letters: '
                  'ICD as I C D, ECG as E C G, ATP as A T P, and S-ICD as S I C D. '
                  'Use the full term if that is clearer. Keep normal abbreviations in visual captions. '
                  'Do not spell out ordinary words or pronounceable acronyms indiscriminately.')

# Plain-text script editing uses a different output contract from JSON exports.
ANIMAKER_NARRATION_RULE = (
    'Voiceover pronunciation is required in every Narration section. Write narration '
    'exactly as it should be spoken, not as technical shorthand. At the first mention, '
    'use the full medication-group name if supplied by the medical draft or sources; '
    'do not invent an expansion. For later mentions, write ARNI as Arni '
    '(and ARNIs as Arnis), a pronounceable word, never as A alone. '
    'Write letter-by-letter initialisms with commas and spaces: MRA as M, R, A; '
    'ICD as I, C, D; ECG as E, C, G; S-ICD as S, I, C, D. '
    'For SGLT2, use S, G, L, T, zwei in German narration and S, G, L, T, two '
    'in English narration. Spell number tokens in the requested spoken language. '
    'For ACE, prefer the supported full name or a plain-language reference to the '
    'medicine group; if using the abbreviation in speech, write A, C, E. '
    'Keep standard abbreviations such as ARNI, MRA and SGLT2 unchanged in '
    '[ON-SCREEN TEXT: ...], titles and visual directions. Do not indiscriminately '
    'spell out ordinary words. If pronunciation is uncertain, use a supported full '
    'name or a clear reference such as this medicine group, translated into the '
    'requested language. Check all narration sections for leftover unexpanded '
    'uppercase medication abbreviations before returning the script. '
)


def spoken_text(text):
    return PATTERN.sub(lambda match: ' '.join(match.group().replace('-', '')), text)


def speech_storyboard(value):
    """Return an independent copy with only narration made suitable for speech."""
    result = copy.deepcopy(value)
    for scene in result['scenes']:
        scene['narration'] = spoken_text(scene['narration'])
    return result
