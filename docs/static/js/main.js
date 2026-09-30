(function () {
  const CASES = {
    1: {
      key: 'case1_brake',
      variants: ['vanilla', 'sam3d'],
      desc: 'Agent<Zone> selects the lead vehicle (front zone, TTC 1.0 s). It brakes hard toward a near rear-end interaction.'
    },
    2: {
      key: 'case2_cutin',
      variants: ['vanilla', 'sam3d'],
      desc: 'A vehicle in the adjacent lane cuts in under short headway (front zone, TTC 1.2 s).'
    },
    3: {
      key: 'case3_drift',
      variants: ['original', 'vanilla', 'sam3d'],
      desc: 'An oncoming vehicle drifts over the center line toward the ego vehicle (front zone, TTC 1.99 s). "Unedited trajectory" renders the recorded path.'
    }
  };

  const video = document.getElementById('case-video');
  const raw = document.getElementById('raw-video');
  const desc = document.getElementById('case-desc');
  const tabs = document.querySelectorAll('.tab');
  const variantBtns = document.querySelectorAll('.variant');
  let current = 1;
  let variant = 'sam3d';

  // The original log follows the generated video, so the two stay frame-aligned.
  function sync() {
    if (Math.abs(raw.currentTime - video.currentTime) > 0.06) raw.currentTime = video.currentTime;
  }
  video.addEventListener('play', () => { sync(); raw.play().catch(() => {}); });
  video.addEventListener('pause', () => raw.pause());
  video.addEventListener('seeked', sync);
  video.addEventListener('timeupdate', sync);
  video.addEventListener('ratechange', () => { raw.playbackRate = video.playbackRate; });

  function load(keepTime) {
    const c = CASES[current];
    if (!keepTime) {
      raw.poster = 'static/videos/' + c.key + '_raw.jpg';
      raw.src = 'static/videos/' + c.key + '_raw.mp4';
      raw.load();
    }
    if (!c.variants.includes(variant)) variant = 'sam3d';
    const t = keepTime ? video.currentTime : 0;
    const base = 'static/videos/' + c.key + '_' + variant;
    video.poster = base + '.jpg';
    video.src = base + '.mp4';
    video.addEventListener('loadedmetadata', function once() {
      video.removeEventListener('loadedmetadata', once);
      if (t && t < video.duration) video.currentTime = t;
      video.play().catch(() => {});
    });
    video.load();

    desc.textContent = c.desc;
    variantBtns.forEach(b => {
      b.hidden = !c.variants.includes(b.dataset.variant);
      b.classList.toggle('active', b.dataset.variant === variant);
    });
  }

  tabs.forEach(tab => tab.addEventListener('click', () => {
    current = Number(tab.dataset.case);
    tabs.forEach(t => {
      const on = t === tab;
      t.classList.toggle('active', on);
      t.setAttribute('aria-selected', on);
    });
    load(false);
  }));

  // Switching variant keeps the playback time so the two renderings are easy to compare.
  variantBtns.forEach(btn => btn.addEventListener('click', () => {
    variant = btn.dataset.variant;
    load(true);
  }));

  desc.textContent = CASES[1].desc;
})();
