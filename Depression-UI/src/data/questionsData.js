export const PHQ8_OPTIONS = [
  { label: 'Not at all', value: 0 },
  { label: 'Several days', value: 1 },
  { label: 'More than half the days', value: 2 },
  { label: 'Nearly every day', value: 3 },
];

export const PHQ8_QUESTIONS = [
  {
    id: 1,
    text: 'Over the last two weeks, how often have you felt little interest or pleasure in doing things?',
    instruction: 'Choose the option that best matches your experience.',
  },
  {
    id: 2,
    text: 'Over the last two weeks, how often have you felt down, depressed, or hopeless?',
    instruction: 'Choose the option that best matches your experience.',
  },
  {
    id: 3,
    text: 'Over the last two weeks, how often have you had trouble falling or staying asleep, or sleeping too much?',
    instruction: 'Choose the option that best matches your experience.',
  },
  {
    id: 4,
    text: 'Over the last two weeks, how often have you felt tired or had little energy?',
    instruction: 'Choose the option that best matches your experience.',
  },
  {
    id: 5,
    text: 'Over the last two weeks, how often have you had a poor appetite or been overeating?',
    instruction: 'Choose the option that best matches your experience.',
  },
  {
    id: 6,
    text: 'Over the last two weeks, how often have you felt bad about yourself, or that you are a failure?',
    instruction: 'Choose the option that best matches your experience.',
  },
  {
    id: 7,
    text: 'Over the last two weeks, how often have you had trouble concentrating on things, such as reading or watching television?',
    instruction: 'Choose the option that best matches your experience.',
  },
  {
    id: 8,
    text: 'Over the last two weeks, how often have you been moving or speaking so slowly that other people have noticed, or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?',
    instruction: 'Choose the option that best matches your experience.',
  },
];

export function buildQuestionSet() {
  return [...PHQ8_QUESTIONS];
}

export function getSeverityLabel(score) {
  if (score <= 4) return 'Minimal';
  if (score <= 9) return 'Mild';
  if (score <= 14) return 'Moderate';
  if (score <= 19) return 'Moderately Severe';
  return 'Severe';
}

export function getSeverityDescription(score) {
  if (score <= 4) return 'Your answers suggest very few current symptoms.';
  if (score <= 9) return 'Your answers suggest mild symptoms that are worth tracking.';
  if (score <= 14) return 'Your answers suggest a moderate symptom burden.';
  if (score <= 19) return 'Your answers suggest a high symptom burden.';
  return 'Your answers suggest a severe symptom burden.';
}

export default {
  PHQ8_OPTIONS,
  PHQ8_QUESTIONS,
  buildQuestionSet,
  getSeverityLabel,
  getSeverityDescription,
};

