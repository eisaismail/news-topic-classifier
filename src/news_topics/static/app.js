const $ = (id) => document.getElementById(id);
const names = {tfidf: 'TF-IDF + logistic regression', minilm: 'MiniLM + logistic regression'};
const percent = (value) => `${(value * 100).toFixed(1)}%`;

async function getJSON(url, options) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Please enter valid article text.');
  return body;
}

function renderModels(result) {
  $('empty').hidden = true;
  const output = $('predictions');
  output.replaceChildren(); output.hidden = false;
  Object.entries(result.models).forEach(([key, model]) => {
    const section = document.createElement('div'); section.className = 'model';
    const label = document.createElement('p'); label.className = 'model-title'; label.textContent = names[key];
    const topic = document.createElement('h3'); topic.className = 'topic'; topic.textContent = model.topic;
    section.append(label, topic);
    model.scores.forEach((item) => {
      const row = document.createElement('div'); row.className = 'score-row';
      const title = document.createElement('span'); title.textContent = item.topic === 'Science and Technology' ? 'Science / Tech' : item.topic;
      const bar = document.createElement('div'); bar.className = 'bar';
      const fill = document.createElement('span'); fill.style.width = percent(item.score); bar.append(fill);
      const value = document.createElement('span'); value.textContent = percent(item.score);
      row.append(title, bar, value); section.append(row);
    });
    output.append(section);
  });
  const agreement = document.createElement('p'); agreement.className = 'agreement';
  agreement.textContent = result.agree ? 'Both models chose the same topic.' : 'The models disagree. This article deserves a closer look.';
  output.append(agreement);
}

$('article').addEventListener('input', () => { $('count').textContent = `${$('article').value.length.toLocaleString()} / 10,000`; });
$('classify-form').addEventListener('submit', async (event) => {
  event.preventDefault(); $('error').hidden = true; $('submit').disabled = true; $('submit').textContent = 'Classifying…';
  try {
    const result = await getJSON('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text:$('article').value})});
    renderModels(result);
  } catch (error) { $('error').textContent = error.message; $('error').hidden = false; }
  finally { $('submit').disabled = false; $('submit').textContent = 'Classify article ↗'; }
});

function selectExample(example) {
  $('article').value = example.text; $('article').dispatchEvent(new Event('input'));
  $('source').textContent = `AG News test row ${example.row_id}. Dataset label: ${example.topic}.${example.kind === 'Model mistake' ? ` Recorded MiniLM prediction: ${example.predicted}.` : ''}`;
  $('classify-form').requestSubmit();
}

async function initialise() {
  try { await getJSON('/api/health'); $('status').textContent = 'Running locally'; $('status').classList.add('ready'); }
  catch { $('status').textContent = 'Models not ready'; }
  try {
    const report = await getJSON('/api/evaluation'); $('metrics').replaceChildren();
    Object.entries(report.test).forEach(([key, result]) => {
      const row = document.createElement('tr');
      [names[key], percent(result.accuracy), result.macro_f1.toFixed(3)].forEach((value) => { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); });
      $('metrics').append(row);
    });
    $('split-note').textContent = `${report.split.training_rows.toLocaleString()} training · ${report.split.validation_rows.toLocaleString()} validation · ${report.split.official_test_rows.toLocaleString()} test articles`;
  } catch { $('metrics').textContent = 'Run training to generate the evaluation.'; }
  try {
    const data = await getJSON('/api/examples');
    data.examples.forEach((example) => {
      const button = document.createElement('button'); button.type = 'button'; button.textContent = example.topic;
      button.addEventListener('click', () => selectExample(example));
      if (example.kind === 'Model mistake') {
        const detail = document.createElement('small'); detail.textContent = `Predicted ${example.predicted}`; button.append(detail); $('mistakes').append(button);
      } else { $('examples').append(button); }
    });
  } catch { $('examples').textContent = 'Download AG News to load examples.'; }
}
initialise();
