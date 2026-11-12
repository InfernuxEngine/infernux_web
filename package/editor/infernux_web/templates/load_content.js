Module.infernuxPresentation = @INFERNUX_WEB_PRESENTATION@;
Module.preRun = [...(Module.preRun || []), function () {
  if (typeof WebAssembly.Suspending !== 'function' || typeof WebAssembly.promising !== 'function') {
    Module.abort('This game requires a browser with WebAssembly Promise Integration. Please update your browser.');
  }
  Module.addRunDependency('infernux-project-content');
  fetch('infernux-player.@INFERNUX_WEB_ASSET_REVISION@.inxpkg')
    .then(response => {
      if (!response.ok) throw new Error(`Game content request failed: HTTP ${response.status}`);
      return response.arrayBuffer();
    })
    .then(content => {
      Module.FS.writeFile('/infernux-project.inxpkg', new Uint8Array(content));
      Module.removeRunDependency('infernux-project-content');
    })
    .catch(error => Module.abort(error.message));
}];
