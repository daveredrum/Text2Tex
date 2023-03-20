window.HELP_IMPROVE_VIDEOJS = false;
window.HELP_IMPROVE_VIDEOJS = false;

let INTERP_BASE = "./static/latent_interpolation/";
let NUM_INTERP_FRAMES = 26;

let interp_images_chair = [];
let interp_images_car = [];
function preloadInterpolationImages() {
    for (let i = 0; i < NUM_INTERP_FRAMES; i++) {
        let path = INTERP_BASE + 'car/' + String(i).padStart(3, '0') + '.jpg';
        interp_images_car[i] = new Image();
        interp_images_car[i].src = path;
        // interp_images_car[i].width = 256;
    }
    for (let i = 0; i < NUM_INTERP_FRAMES; i++) {
        let path = INTERP_BASE + 'chair/' + String(i).padStart(3, '0') + '.jpg';
        interp_images_chair[i] = new Image();
        interp_images_chair[i].src = path;
        // interp_images_chair[i].width = 256;
    }
}

function setInterpolationImageChair(i) {
  let image_chair = interp_images_chair[i];
  image_chair.ondragstart = function() { return false; };
  image_chair.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper-chair').empty().append(image_chair);
}

function setInterpolationImageCar(i) {
  let image_car = interp_images_car[i];
  image_car.ondragstart = function() { return false; };
  image_car.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper-car').empty().append(image_car);
}

$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    let options = {
			slidesToScroll: 1,
			slidesToShow: 4,
			loop: false,
			infinite: false,
            pagination: false,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    let carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(let i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    let element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*
    let player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);
    */

    preloadInterpolationImages();

    $('#interpolation-slider-chair').on('input', function(event) {
      setInterpolationImageChair(this.value);
    });
    setInterpolationImageChair(0);
    $('#interpolation-slider-car').on('input', function(event) {
      setInterpolationImageCar(this.value);
    });
    setInterpolationImageCar(0);

    $('#interpolation-slider-chair').prop('max', NUM_INTERP_FRAMES - 1);
    $('#interpolation-slider-car').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();
})
