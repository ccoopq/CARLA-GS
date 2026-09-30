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
      desc: 'An oncoming vehicle drifts over the center line toward the ego vehicle (front zone, TTC 1.99 s). "Unedited path" renders the recorded trajectory.'
    }
  };
  const BADGE = { original: 'CARLA-GS · unedited path', vanilla: 'CARLA-GS · vanilla 3DGS', sam3d: 'CARLA-GS · SAM 3D' };

  const raw = document.getElementById('raw-video');
  const genFrame = document.getElementById('gen-frame');
  const badge = document.getElementById('gen-badge');
  const desc = document.getElementById('case-desc');
  const tabs = document.querySelectorAll('.tab');
  const variantBtns = document.querySelectorAll('.variant');
  let current = 1;
  let variant = 'sam3d';
  let gens = {};

  // Every variant of the current case is loaded and kept in sync with the original log,
  // so switching variant only changes which video is visible.
  function buildCase() {
    const c = CASES[current];
    if (!c.variants.includes(variant)) variant = 'sam3d';
    Object.values(gens).forEach(v => v.remove());
    gens = {};
    c.variants.forEach(name => {
      const v = document.createElement('video');
      v.muted = true; v.loop = true; v.playsInline = true; v.preload = 'auto';
      v.poster = 'static/videos/' + c.key + '_' + name + '.jpg';
      v.src = 'static/videos/' + c.key + '_' + name + '.mp4';
      genFrame.insertBefore(v, badge);
      gens[name] = v;
    });
    raw.poster = 'static/videos/' + c.key + '_raw.jpg';
    raw.src = 'static/videos/' + c.key + '_raw.mp4';
    raw.play().catch(() => {});
    desc.textContent = c.desc;
    showVariant();
  }

  function showVariant() {
    const c = CASES[current];
    Object.entries(gens).forEach(([name, v]) => { v.hidden = name !== variant; });
    badge.textContent = BADGE[variant];
    variantBtns.forEach(b => {
      b.hidden = !c.variants.includes(b.dataset.variant);
      b.classList.toggle('active', b.dataset.variant === variant);
    });
    sync();
  }

  function sync() {
    Object.values(gens).forEach(v => {
      if (v.readyState >= 1 && Math.abs(v.currentTime - raw.currentTime) > 0.06) v.currentTime = raw.currentTime;
      if (raw.paused !== v.paused) (raw.paused ? v.pause() : v.play().catch(() => {}));
    });
  }
  ['timeupdate', 'play', 'pause', 'seeked'].forEach(e => raw.addEventListener(e, sync));

  document.getElementById('compare').addEventListener('click', () => {
    raw.paused ? raw.play().catch(() => {}) : raw.pause();
  });

  tabs.forEach(tab => tab.addEventListener('click', () => {
    current = Number(tab.dataset.case);
    tabs.forEach(t => {
      const on = t === tab;
      t.classList.toggle('active', on);
      t.setAttribute('aria-selected', on);
    });
    buildCase();
  }));

  variantBtns.forEach(btn => btn.addEventListener('click', () => {
    variant = btn.dataset.variant;
    showVariant();
  }));

  buildCase();
})();
