(function () {
  const CASES = {
    1: {
      key: 'case1_brake',
      variants: ['vanilla', 'sam3d']
    },
    2: {
      key: 'case2_cutin',
      variants: ['vanilla', 'sam3d']
    },
    3: {
      key: 'case3_drift',
      variants: ['original', 'vanilla', 'sam3d']
    }
  };
  const BADGE = { original: 'CARLA-GS · unedited path', vanilla: 'CARLA-GS · vanilla 3DGS', sam3d: 'CARLA-GS · SAM 3D' };

  const video = document.getElementById('case-video');
  const raw = document.getElementById('raw-video');
  const badge = document.getElementById('gen-badge');
  const tabs = document.querySelectorAll('.tab');
  const variantBtns = document.querySelectorAll('.variant');
  let current = 1;
  let variant = 'sam3d';
  let request = 0;

  // Only two videos ever decode at once. Other variants are downloaded as plain data,
  // so switching to them does not wait on the network.
  const blobs = {};
  function prefetch(url) {
    if (!blobs[url]) {
      blobs[url] = fetch(url).then(r => r.ok ? r.blob() : Promise.reject())
        .then(b => URL.createObjectURL(b)).catch(() => url);
    }
    return blobs[url];
  }
  const url = (key, name) => 'static/videos/' + key + '_' + name + '.mp4';

  // The original log follows the generated video, so the two stay frame-aligned.
  function sync() {
    if (Math.abs(raw.currentTime - video.currentTime) > 0.06) raw.currentTime = video.currentTime;
  }
  video.addEventListener('play', () => { sync(); raw.play().catch(() => {}); });
  video.addEventListener('pause', () => raw.pause());
  video.addEventListener('seeked', sync);
  video.addEventListener('timeupdate', sync);

  function setGen(src, t) {
    video.addEventListener('loadedmetadata', function once() {
      video.removeEventListener('loadedmetadata', once);
      if (t && t < video.duration) video.currentTime = t;
      video.play().catch(() => {});
    });
    video.src = src;
    video.load();
  }

  function load(keepTime) {
    const c = CASES[current];
    const id = ++request;
    if (!c.variants.includes(variant)) variant = 'sam3d';
    const t = keepTime ? video.currentTime : 0;
    video.poster = 'static/videos/' + c.key + '_' + variant + '.jpg';
    if (keepTime) {
      prefetch(url(c.key, variant)).then(src => { if (id === request) setGen(src, t); });
    } else {
      raw.poster = 'static/videos/' + c.key + '_raw.jpg';
      raw.src = url(c.key, 'raw');
      raw.load();
      setGen(url(c.key, variant), 0);
      c.variants.forEach(name => { if (name !== variant) prefetch(url(c.key, name)); });
    }
    badge.textContent = BADGE[variant];
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

  // Switching variant keeps the playback time so the renderings are easy to compare.
  variantBtns.forEach(btn => btn.addEventListener('click', () => {
    variant = btn.dataset.variant;
    load(true);
  }));

  // Browsers may hold back autoplay for videos that start off screen; start it once visible.
  new IntersectionObserver(entries => {
    if (entries.some(e => e.isIntersecting) && video.paused) video.play().catch(() => {});
  }, { threshold: 0.25 }).observe(video);

  CASES[1].variants.forEach(name => { if (name !== variant) prefetch(url(CASES[1].key, name)); });
})();
