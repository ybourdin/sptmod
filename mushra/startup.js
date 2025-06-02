/*************************************************************************
         (C) Copyright AudioLabs 2017

This source code is protected by copyright law and international treaties. This source code is made available to You subject to the terms and conditions of the Software License for the webMUSHRA.js Software. Said terms and conditions have been made available to You prior to Your download of this source code. By downloading this source code You agree to be bound by the above mentionend terms and conditions, which can also be found here: https://www.audiolabs-erlangen.de/resources/webMUSHRA. Any unauthorised use of this source code may result in severe civil and criminal penalties, and will be prosecuted to the maximum extent possible under law.

**************************************************************************/

/**
 * Minified by jsDelivr using Terser v5.19.2.
 * Original file: /npm/fingerprintjs@0.5.3/fingerprint.js
 *
 * Do NOT use SRI with dynamically generated files! More information: https://www.jsdelivr.com/using-sri-with-dynamic-files
 */
!function(t,e,n){"undefined"!=typeof module&&module.exports?module.exports=n():"function"==typeof define&&define.amd?define(n):e.Fingerprint=n()}(0,this,(function(){"use strict";var t=function(t){var e,n;e=Array.prototype.forEach,n=Array.prototype.map,this.each=function(t,n,r){if(null!==t)if(e&&t.forEach===e)t.forEach(n,r);else if(t.length===+t.length){for(var i=0,a=t.length;i<a;i++)if(n.call(r,t[i],i,t)==={})return}else for(var o in t)if(t.hasOwnProperty(o)&&n.call(r,t[o],o,t)==={})return},this.map=function(t,e,r){var i=[];return null==t?i:n&&t.map===n?t.map(e,r):(this.each(t,(function(t,n,a){i[i.length]=e.call(r,t,n,a)})),i)},"object"==typeof t?(this.hasher=t.hasher,this.screen_resolution=t.screen_resolution,this.canvas=t.canvas,this.ie_activex=t.ie_activex):"function"==typeof t&&(this.hasher=t)};return t.prototype={get:function(){var t=[];(t.push(navigator.userAgent),t.push(navigator.language),t.push(screen.colorDepth),this.screen_resolution)&&(void 0!==this.getScreenResolution()&&t.push(this.getScreenResolution().join("x")));return t.push((new Date).getTimezoneOffset()),t.push(this.hasSessionStorage()),t.push(this.hasLocalStorage()),t.push(!!window.indexedDB),document.body?t.push(typeof document.body.addBehavior):t.push("undefined"),t.push(typeof window.openDatabase),t.push(navigator.cpuClass),t.push(navigator.platform),t.push(navigator.doNotTrack),t.push(this.getPluginsString()),this.canvas&&this.isCanvasSupported()&&t.push(this.getCanvasFingerprint()),this.hasher?this.hasher(t.join("###"),31):this.murmurhash3_32_gc(t.join("###"),31)},murmurhash3_32_gc:function(t,e){var n,r,i,a,o,s,u,c;for(n=3&t.length,r=t.length-n,i=e,o=3432918353,s=461845907,c=0;c<r;)u=255&t.charCodeAt(c)|(255&t.charCodeAt(++c))<<8|(255&t.charCodeAt(++c))<<16|(255&t.charCodeAt(++c))<<24,++c,i=27492+(65535&(a=5*(65535&(i=(i^=u=(65535&(u=(u=(65535&u)*o+(((u>>>16)*o&65535)<<16)&4294967295)<<15|u>>>17))*s+(((u>>>16)*s&65535)<<16)&4294967295)<<13|i>>>19))+((5*(i>>>16)&65535)<<16)&4294967295))+((58964+(a>>>16)&65535)<<16);switch(u=0,n){case 3:u^=(255&t.charCodeAt(c+2))<<16;case 2:u^=(255&t.charCodeAt(c+1))<<8;case 1:i^=u=(65535&(u=(u=(65535&(u^=255&t.charCodeAt(c)))*o+(((u>>>16)*o&65535)<<16)&4294967295)<<15|u>>>17))*s+(((u>>>16)*s&65535)<<16)&4294967295}return i^=t.length,i=2246822507*(65535&(i^=i>>>16))+((2246822507*(i>>>16)&65535)<<16)&4294967295,i=3266489909*(65535&(i^=i>>>13))+((3266489909*(i>>>16)&65535)<<16)&4294967295,(i^=i>>>16)>>>0},hasLocalStorage:function(){try{return!!window.localStorage}catch(t){return!0}},hasSessionStorage:function(){try{return!!window.sessionStorage}catch(t){return!0}},isCanvasSupported:function(){var t=document.createElement("canvas");return!(!t.getContext||!t.getContext("2d"))},isIE:function(){return"Microsoft Internet Explorer"===navigator.appName||!("Netscape"!==navigator.appName||!/Trident/.test(navigator.userAgent))},getPluginsString:function(){return this.isIE()&&this.ie_activex?this.getIEPluginsString():this.getRegularPluginsString()},getRegularPluginsString:function(){return this.map(navigator.plugins,(function(t){var e=this.map(t,(function(t){return[t.type,t.suffixes].join("~")})).join(",");return[t.name,t.description,e].join("::")}),this).join(";")},getIEPluginsString:function(){if(window.ActiveXObject){return this.map(["ShockwaveFlash.ShockwaveFlash","AcroPDF.PDF","PDF.PdfCtrl","QuickTime.QuickTime","rmocx.RealPlayer G2 Control","rmocx.RealPlayer G2 Control.1","RealPlayer.RealPlayer(tm) ActiveX Control (32-bit)","RealVideo.RealVideo(tm) ActiveX Control (32-bit)","RealPlayer","SWCtl.SWCtl","WMPlayer.OCX","AgControl.AgControl","Skype.Detection"],(function(t){try{return new ActiveXObject(t),t}catch(t){return null}})).join(";")}return""},getScreenResolution:function(){return[screen.height,screen.width]},getCanvasFingerprint:function(){var t=document.createElement("canvas"),e=t.getContext("2d"),n="http://valve.github.io";return e.textBaseline="top",e.font="14px 'Arial'",e.textBaseline="alphabetic",e.fillStyle="#f60",e.fillRect(125,1,62,20),e.fillStyle="#069",e.fillText(n,2,15),e.fillStyle="rgba(102, 204, 0, 0.7)",e.fillText(n,4,17),t.toDataURL()}},t}));
//# sourceMappingURL=/sm/1f7a46c87aaf8e080e3cb4075d74d7e93e22f2e2d1269bfbd2371d9829c2b2e6.map

function checkOrientation() {//when changing from potrait to landscape change to the rigth width

  var siteWidth = document.body.scrollWidth;
  $("#header").css("width", siteWidth.toString());

}

window.onresize = function(event) {
  if (pageManager.getCurrentPage() && pageManager.getCurrentPage().isMushra == true) {
    pageManager.getCurrentPage().renderCanvas("mushra_items");
  }

  checkOrientation();
};

// $(document).ready(function(){
// $(window).scroll(function(){
// $('#header').css({
// 'left': $(this).scrollLeft()//Note commented because it causes the endless scrolling to the left
// });
// });
// });


// callbacks
function callbackFilesLoaded() {
  pageManager.start();
  pageTemplateRenderer.renderProgressBar(("page_progressbar"));
  pageTemplateRenderer.renderHeader(("page_header"));
  pageTemplateRenderer.renderNavigation(("page_navigation"));

  if (config.stopOnErrors == false || !errorHandler.errorOccurred()) {
    $.mobile.loading("hide");
    $("body").children().children().removeClass('ui-disabled');
  } else {
    var errors = errorHandler.getErrors();
    var ul = $("<ul style='text-align:left;'></ul>");
    $('#popupErrorsContent').append(ul);
    for (var i = 0; i < errors.length; ++i) {
      ul.append($('<li>' + errors[i] + '</li>'));
    }
    $("#popupErrors").popup("open");
    $.mobile.loading("hide");
  }

  if ($.mobile.activePage) {
    $.mobile.activePage.trigger('create');
  }
}

function callbackURLFound() {
  var errors = errorHandler.getErrors();
  var ul = $("<ul style='text-align:left;'></ul>");
  $('#popupErrorsContent').append(ul);
  for (var i = 0; i < errors.length; ++i) {
    ul.append($('<li>' + errors[i] + '</li>'));
  }
  $("#popupErrors").popup("open");
}

function addPagesToPageManager(_pageManager, _pages) {
  for (var i = 0; i < _pages.length; ++i) {
    if (Array.isArray(_pages[i])) {
      if (_pages[i][0] === "random") {
        _pages[i].shift();
        shuffle(_pages[i]);
      }
      addPagesToPageManager(_pageManager, _pages[i]);
    } else {
      var pageConfig = _pages[i];
      if (pageConfig.type == "generic") {
        _pageManager.addPage(new GenericPage(_pageManager, pageConfig));
      } else if (pageConfig.type == "consent") {
        _pageManager.addPage(new ConsentPage(_pageManager, pageTemplateRenderer, pageConfig));
      } else if (pageConfig.type == "volume") {
        var volumePage = new VolumePage(_pageManager, audioContext, audioFileLoader, pageConfig, config.bufferSize, errorHandler, config.language);
        _pageManager.addPage(volumePage);
      } else if (pageConfig.type == "mushra") {
        var mushraPage = new MushraPage(_pageManager, audioContext, config.bufferSize, audioFileLoader, session, pageConfig, mushraValidator, errorHandler, config.language);
        _pageManager.addPage(mushraPage);
      } else if ( pageConfig.type == "spatial"){
        _pageManager.addPage(new SpatialPage(_pageManager, pageConfig, session, audioContext, config.bufferSize, audioFileLoader, errorHandler, config.language));
      } else if (pageConfig.type == "paired_comparison") {
        var pcPageManager = new PairedComparisonPageManager();
        pcPageManager.createPages(_pageManager, pageTemplateRenderer, pageConfig, audioContext, config.bufferSize, audioFileLoader, session, errorHandler, config.language);
        pcPageManager = null;
      } else if (pageConfig.type == "bs1116") {
        var bs1116PageManager = new BS1116PageManager();
        bs1116PageManager.createPages(_pageManager, pageTemplateRenderer, pageConfig, audioContext, config.bufferSize, audioFileLoader, session, errorHandler, config.language);
        bs1116PageManager = null;
      } else if (pageConfig.type == "likert_single_stimulus") {
        var likertSingleStimulusPageManager = new LikertSingleStimulusPageManager();
        likertSingleStimulusPageManager.createPages(_pageManager, pageTemplateRenderer, pageConfig, audioContext, config.bufferSize, audioFileLoader, session, errorHandler, config.language);
        likertSingleStimulusPageManager = null;
      } else if (pageConfig.type == "likert_multi_stimulus") {
        var likertMultiStimulusPage = new LikertMultiStimulusPage(pageManager, pageTemplateRenderer, pageConfig, audioContext, config.bufferSize, audioFileLoader, session, errorHandler, config.language);
        _pageManager.addPage(likertMultiStimulusPage);
      } else if (pageConfig.type == "finish") {
        var finishPage = new FinishPage(_pageManager, session, dataSender, pageConfig, config.language);
        _pageManager.addPage(finishPage);
      } else {

        errorHandler.sendError("Type not specified.");

      }
    }
  }
}

for (var i = 0; i < $("body").children().length; i++) {
  if ($("body").children().eq(i).attr('id') != "popupErrors" && $("body").children().eq(i).attr('id') != "popupDialog") {
    $("body").children().eq(i).addClass('ui-disabled');
  }
}




function startup(config) {


  if (config == null) {
    errorHandler.sendError("URL couldn't be found!");
    callbackURLFound();
  }

  $.mobile.page.prototype.options.theme = 'a';
  var interval = setInterval(function() {
    $.mobile.loading("show", {
      text : "Loading...",
      textVisible : true,
      theme : "a",
      html : ""
    });
    clearInterval(interval);
  }, 1);
  
  
  if (pageManager !== null) { // clear everything for new experiment
    pageTemplateRenderer.clear();
    $("#page_content").empty();
    $('#header').empty();
  }

  localizer = new Localizer();
  localizer.initializeNLSFragments(nls);

  pageManager = null;
  audioContext;
  audioFileLoader = null;
  mushraValidator = null;
  dataSender = null;
  session = null;
  pageTemplateRenderer = null;
  interval2 = null;

  document.title = config.testname;
  $('#header').append(document.createTextNode(config.testname));

  pageManager = new PageManager("pageManager", "page_content", localizer);
  window.AudioContext = window.AudioContext || window.webkitAudioContext;

  if ( typeof AudioContext !== 'undefined') {
    audioContext = new AudioContext();
  } else if ( typeof webkitAudioContext !== 'undefined') {
    audioContext = new webkitAudioContext();
  }

  document.addEventListener("click", function () {
    if (audioContext.state !== 'running') {
      audioContext.resume();
    }
  }, true);

  try {
    audioContext.destination.channelCountMode = "explicit";
    audioContext.destination.channelInterpretation = "discrete";
    audioContext.destination.channelCount = audioContext.destination.maxChannelCount;
  } catch (e) {
    console.log("webMUSHRA: Could not set channel count of destination node.");
    console.log(e);
  }
  audioContext.volume = 1.0;

  audioFileLoader = new AudioFileLoader(audioContext, errorHandler);
  mushraValidator = new MushraValidator(errorHandler);
  dataSender = new DataSender(config);

  session = new Session();
  session.testId = config.testId;
  session.config = configFile;

  if (config.language == undefined) {
    config.language = 'en';
  }
  pageTemplateRenderer = new PageTemplateRenderer(pageManager, config.showButtonPreviousPage, config.language);
  pageManager.addCallbackPageEventChanged(pageTemplateRenderer.refresh.bind(pageTemplateRenderer));

  addPagesToPageManager(pageManager, config.pages);

  interval2 = setInterval(function() {
    clearInterval(interval2);
    audioFileLoader.startLoading(callbackFilesLoaded);
  }, 10);

}

// start code (loads config) 

function getParameterByName(name) {
  var match = RegExp('[?&]' + name + '=([^&]*)').exec(window.location.search);
  return match && decodeURIComponent(match[1].replace(/\+/g, ' '));
}

var config = null;
var configArg = getParameterByName("config");
var configFile = '';
if (configArg) {
  configFile = 'configs/' + configArg;
} else {
  const fingerprint = new Fingerprint({canvas: true}).get();
  const fingerprint2 = new Fingerprint({canvas: false}).get();
  // configFile = 'configs/default.yaml';
  const a = fingerprint % 2;
  const b = fingerprint2 % 10;
  configFile = `configs/conf25_${a}_${b}.yaml`
  console.log("to be loaded", configFile)
}


// global variables
var errorHandler = new ErrorHandler();
var localizer = null;
var pageManager = null;
var audioContext = null;
var audioFileLoader = null;
var mushraValidator = null;
var dataSender = null;
var session = null;
var pageTemplateRenderer = null;
var interval2 = null;


YAML.load(configFile, (function(result) {
  config = result;
  startup(result);
}));
