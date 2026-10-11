(() => {
  'use strict';
  const slides = [...document.querySelectorAll('.slide')];
  const select = document.querySelector('#contents');
  const previous = document.querySelector('#previous');
  const next = document.querySelector('#next');
  const explain = document.querySelector('#explain');
  const read = document.querySelector('#read');
  const progress = document.querySelector('#progress');
  const announce = document.querySelector('#announce');
  let current = Math.max(0, slides.findIndex(slide => slide.id === location.hash.slice(1)));
  let reading = false;
  for (const [i, slide] of slides.entries()) {
    const option = document.createElement('option');
    option.value = String(i);
    option.textContent = `${i + 1}. ${slide.dataset.title}`;
    select.append(option);
  }
  function show(index, updateHash = true, focus = false) {
    current = Math.max(0, Math.min(slides.length - 1, index));
    for (const [i, slide] of slides.entries()) slide.classList.toggle('active', i === current);
    select.value = String(current);
    previous.disabled = current === 0;
    next.disabled = current === slides.length - 1;
    progress.textContent = `${current + 1} / ${slides.length}`;
    explain.setAttribute('aria-pressed', String(slides[current].classList.contains('notes-open')));
    announce.textContent = `Slide ${current + 1} of ${slides.length}: ${slides[current].dataset.title}`;
    if (updateHash) history.replaceState(null, '', '#' + slides[current].id);
    if (reading) slides[current].scrollIntoView({block:'start'});
    else window.scrollTo(0, 0);
    if (focus) slides[current].querySelector('h1,h2').focus({preventScroll:true});
  }
  previous.addEventListener('click', () => show(current - 1));
  next.addEventListener('click', () => show(current + 1));
  select.addEventListener('change', () => show(Number(select.value), true, true));
  explain.addEventListener('click', () => {
    const open = !slides[current].classList.contains('notes-open');
    slides[current].classList.toggle('notes-open', open);
    slides[current].querySelector('details').open = open;
    explain.setAttribute('aria-pressed', String(open));
  });
  read.addEventListener('click', () => {
    reading = !reading;
    document.body.classList.toggle('reading', reading);
    read.textContent = reading ? 'Slide view' : 'Read all';
    read.setAttribute('aria-pressed', String(reading));
    slides[current].scrollIntoView({block:'start'});
  });
  document.querySelector('#print').addEventListener('click', () => window.print());
  document.addEventListener('keydown', event => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.target.closest('input,select,textarea,button,a,summary,[contenteditable]')) return;
    if (reading) return;
    if (['ArrowRight','PageDown','ArrowLeft','PageUp','Home','End'].includes(event.key)) event.preventDefault();
    if (['ArrowRight','PageDown'].includes(event.key)) show(current + 1);
    if (['ArrowLeft','PageUp'].includes(event.key)) show(current - 1);
    if (event.key === 'Home') show(0);
    if (event.key === 'End') show(slides.length - 1);
  });
  window.addEventListener('hashchange', () => {
    const index = slides.findIndex(slide => slide.id === location.hash.slice(1));
    if (index >= 0) show(index, false);
  });
  document.body.classList.add('enhanced');
  show(current, false);
})();
