/// タグリストの読み出し
var _taglist = null;
$.ajax({
    url: './api/taglist.json',
    dataType: 'json',
    async: false,
    success: function (data) {
        _taglist = data;
    }
});
var _taglist_jp = null;
$.ajax({
    url: './static/taglist_jp.json',
    dataType: 'json',
    async: false,
    success: function (data) {
        _taglist_jp = data;
    }
});

/// 設定
$.cookie.json = true;

/// keras jsの準備
var model_kerasjs;
var worker_kerasjs;
if (generate_image_mode == 'keras_js') {
    model_kerasjs = new KerasJS.Model({
        filepaths: {
            model: "./api/kerasjs/keras_generator_arch.json",
            weights: "./api/kerasjs/keras_generator_weights.buf",
            metadata: "./api/kerasjs/keras_generator_metadata.json",
        },
        gpu: true,
    });

    // worker代わり。pushする関数の最後に setTimeout(worker, 200); などと書いてループを回す
    worker_kerasjs = [];
    function worker() {
        if (worker_kerasjs.length > 0) {
            console.log("predicting...");
            worker_kerasjs[0]();
            worker_kerasjs.shift();
        }
        else {
            setTimeout(worker, 200);
        }
    }

    $(document).ready(function () {
        console.log("model loading...");
        $.blockUI({
            message: '<img src="./static/image/model_loading.gif" style="margin-right: 10px">Model Loading...',
            showOverlay: false,
            css: {
                width: '350px', top: '10px', left: '', right: '10px',
                border: 'none', padding: '5px', backgroundColor: '#000', opacity: .6, color: '#fff',
                '-webkit-border-radius': '10px', '-moz-border-radius': '10px',
            }
        });

        model_kerasjs.ready().then(function () {
            console.log("keras model prepared!");
            $.unblockUI();
            worker();
        });
    });

}

// image idとtagやseed値の情報のペア
var info_each_image = {};


/**
 * タグリスト取得
 */
function getTagList() {
    return _taglist;
}
function getTagListJP() {
    return _taglist_jp;
}

/**
 * 生成画像を表示するためのimgを作成
 */
function createGeneratedImageElement() {
    var elem;
    if (generate_image_mode == 'server') {
        elem = document.createElement("img");
    }
    else if (generate_image_mode == 'keras_js') {
        elem = document.createElement('canvas');
        elem.setAttribute("width", 96);
        elem.setAttribute("height", 96);
    }
    else {
        throw new Error("unknown generate_image_mode: " + generate_image_mode)
    }

    elem.className = 'generated';
    elem.id = 'id' + Math.random().toString(36).substring(8);
    return elem;
}

/**
 * 非同期でkerasjsを利用して画像生成してImageDataを置換
 */
function replaceImageDataAsync(canvas_or_id, input_tag, input_z) {
    const inputData = {
        'input_z': new Float32Array(input_z),
        'input_tag': new Float32Array(input_tag),
    };

    worker_kerasjs.push(function () {
        model_kerasjs.predict(inputData).then(function (outputData) {
            var output_tensor = outputData.float_m1p1;

            var imgarray = new Uint8Array(output_tensor.data.map(function (x) {
                return (x + 1) / 2 * 255;
            }));
            var imageData = image2Darray(imgarray, output_tensor.shape[0], output_tensor.shape[1]);

            var canvas = (typeof(canvas_or_id) === 'string') ? $('#' + canvas_or_id)[0] : canvas_or_id;
            if (canvas) {
                var context = canvas.getContext('2d');
                context.putImageData(imageData, 0, 0);
            }

            setTimeout(worker, 200);
        });
    });
}

/**
 * 要素に生成画像に関する情報を付加
 */
function putGeneratedImageAttribute(elem_or_id, tags, seed) {
    if (generate_image_mode == 'server') {
        var elem = (typeof(elem_or_id) === 'string') ? $('#' + elem_or_id)[0] : elem_or_id;
        var str_json = JSON.stringify(tags);
        elem.src = "./api/make_image_fix_z" + "?tags=" + str_json + "&seed=" + seed;
    }
    else if (generate_image_mode == 'keras_js') {
        var input_tag = convertTagsToTagBinary(tags);
        var input_z = convertSeedToRandn(seed);
        replaceImageDataAsync(elem_or_id, input_tag, input_z)
    }

    var id = (typeof(elem_or_id) === 'string') ? elem_or_id : elem_or_id.id;
    info_each_image[id] = {
        'tags': tags,
        'seed': seed,
    };
}

/**
 * 要素にモーフィング画像に関する情報を付加
 */
function putMorphingImagesAttribute(elem_or_id_list, tags1, tags2, seed) {
    if (generate_image_mode == 'server') {
        elem_or_id_list.forEach(function (elem_or_id, i) {
            var elem = (typeof(elem_or_id) === 'string') ? $('#' + elem_or_id)[0] : elem_or_id;
            var str_json1 = JSON.stringify(tags1);
            var str_json2 = JSON.stringify(tags2);
            elem.src = "./api/make_image_morphing"
                + "?tags1=" + str_json1
                + "&tags2=" + str_json2
                + "&num_stage=" + elem_or_id_list.length
                + "&i_stage=" + i
                + "&seed=" + seed;
        });
    }
    else if (generate_image_mode == 'keras_js') {
        var embeddings = convertTagsToTagEmbedding(tags1, tags2, elem_or_id_list.length);
        var input_z = convertSeedToRandn(seed);
        elem_or_id_list.forEach(function (elem_or_id, i) {
            replaceImageDataAsync(elem_or_id, embeddings[i], input_z);
        });
    }
}

/**
 * 3次元のFloat32Arrayを4次元のImageDataに変換
 */
function image2Darray(imgarray, width, height) {
    const size = width * height * 4;
    var imageData = new Uint8ClampedArray(size);
    for (var i = 0; i < size; i++) {
        imageData[i * 4 + 0] = imgarray[i * 3 + 0];
        imageData[i * 4 + 1] = imgarray[i * 3 + 1];
        imageData[i * 4 + 2] = imgarray[i * 3 + 2];
        imageData[i * 4 + 3] = 255;
    }
    return new ImageData(imageData, width, height)
}

/**
 * 生成画像を右クリック可能にする
 */
function registImageToContextMenu() {
    $('.generated').contextmenu({
        target: '#context-menu',
        onItem: function (context, e) {
            var elem = context[0];
            var seed = info_each_image[elem.id]['seed'];
            var tags = info_each_image[elem.id]['tags'];

            var registered = $.cookie('registered');
            if (registered == null) registered = [];
            registered.unshift({seed: seed, tags: tags});
            registered = registered.slice(0, 7);
            $.cookie('registered', registered, {path: '/'})
        }
    });
}

/**
 * 登録したデータを取得する
 */
function getRegistered() {
    return $.cookie()['registered'];
}

/**
 * 選択されているタグ一覧を取得。
 * `str_start_id`+`タグ名`のidが振られているcheckboxを総なめする。
 * @param str_start_id idの先頭文字
 */
function getSelectedTagList(str_start_id) {
    var checked_tags = [];
    _taglist.forEach(function (tag) {
        var checkbox = document.getElementById('tag_check_' + tag);
        if (checkbox == null) {
            return;
        }
        if (checkbox.checked) {
            checked_tags.push(tag);
        }
    });
    return checked_tags;
}

/**
 * ランダムなシード文字列を取得
 */
function getRandomSeed() {
    var num_z = 100;
    var z = (new Array(num_z)).fill(0).map(function (o) {
        return randn(0, 1);
    });

    return convertRandnToSeed(z);
}

/**
 * シード値から乱数列を取得
 */
function convertSeedToRandn(seed) {
    var seed_base64 = seed.replace(/-/g, '+').replace(/_/g, '/');
    var bin_seed = atob(seed_base64);
    var array = [];
    for (var i = 0; i < bin_seed.length / 2; i++) {
        array.push(decodeFloat16(bin_seed.charCodeAt(i * 2) | bin_seed.charCodeAt(i * 2 + 1) << 8))
    }
    return array
}

/**
 * 乱数列からシード値を取得
 */
function convertRandnToSeed(z) {
    var int16array = new Uint16Array(z.length);
    z.forEach(function (o, i) {
        int16array[i] = encodeFloat16(o)
    });
    var seed_base64 = btoa(String.fromCharCode.apply(null, new Uint8Array(int16array.buffer)));
    var seed = seed_base64.replace(/\+/g, '-').replace(/\//g, '_');
    return seed;
}

/**
 * タグ文字列からタグバイナリ列を取得
 */
function convertTagsToTagBinary(tags) {
    var binary = (new Array(_taglist.length)).fill(0);
    tags.forEach(function (o) {
        var tagid = _taglist.indexOf(o);
        binary[tagid] = 1;
    });
    return binary
}

/**
 * タグ文字列からモーフィング用のタグembeddingを取得
 */
function convertTagsToTagEmbedding(tags1, tags2, num_stage) {
    var tagbinary1 = convertTagsToTagBinary(tags1);
    var tagbinary2 = convertTagsToTagBinary(tags2);

    var list_embedding = [];
    for (var i_stage = 0; i_stage < num_stage; i_stage++) {
        var embedding = new Array(_taglist.length);
        for (var i = 0; i < _taglist.length; i++) {
            embedding[i] = ((num_stage - 1 - i_stage) * tagbinary1[i] + i_stage * tagbinary2[i]) / (num_stage - 1);
        }
        list_embedding.push(embedding);
    }
    return list_embedding;
}

/**
 * モーフィングするシード文字列を取得
 */
function getMorphingSeeds(seed1, seed2, num_stage) {
    var rand1 = convertSeedToRandn(seed1);
    var rand2 = convertSeedToRandn(seed2);

    var list_rand = [];
    for (var i_stage = 0; i_stage < num_stage; i_stage++) {
        var rand = new Array(rand1.length);
        for (var i = 0; i < rand1.length; i++) {
            rand[i] = ((num_stage - 1 - i_stage) * rand1[i] + i_stage * rand2[i]) / (num_stage - 1);
        }
        list_rand.push(rand);
    }

    var list_seed = list_rand.map(function (z) {
        return convertRandnToSeed(z)
    });
    return list_seed;
}

// http://stackoverflow.com/questions/5678432/decompressing-half-precision-floats-in-javascript
function decodeFloat16(binary) {
    var exponent = (binary & 0x7C00) >> 10;
    var fraction = binary & 0x03FF;
    return (binary >> 15 ? -1 : 1) * (
            exponent ?
                (
                    exponent === 0x1F ?
                        fraction ? NaN : Infinity :
                    Math.pow(2, exponent - 15) * (1 + fraction / 0x400)
                ) :
            6.103515625e-5 * (fraction / 0x400)
        );
}

// http://stackoverflow.com/questions/6162651/half-precision-floating-point-in-java/6162687#6162687
function encodeFloat16(fval) {
    var floatView = new Float32Array(1);
    var int32View = new Uint32Array(floatView.buffer);

    floatView[0] = fval;
    var fbits = int32View[0];
    var sign = (fbits >> 16) & 0x8000;          // sign only
    var val = ( fbits & 0x7fffffff ) + 0x1000; // rounded value

    if (val >= 0x47800000) {             // might be or become NaN/Inf
        if (( fbits & 0x7fffffff ) >= 0x47800000) {
            // is or must become NaN/Inf
            if (val < 0x7f800000) {          // was value but too large
                return sign | 0x7c00;           // make it +/-Inf
            }
            return sign | 0x7c00 |            // remains +/-Inf or NaN
                ( fbits & 0x007fffff ) >> 13; // keep NaN (and Inf) bits
        }
        return sign | 0x7bff;               // unrounded not quite Inf
    }
    if (val >= 0x38800000) {             // remains normalized value
        return sign | val - 0x38000000 >> 13; // exp - 127 + 15
    }
    if (val < 0x33000000) {             // too small for subnormal
        return sign;                        // becomes +/-0
    }
    val = ( fbits & 0x7fffffff ) >> 23;   // tmp exp for subnormal calc
    return sign | ( ( fbits & 0x7fffff | 0x800000 ) // add subnormal bit
        + ( 0x800000 >>> val - 102 )     // round depending on cut off
        >> 126 - val );                  // div by 2^(1-(exp-127+15)) and >> 13 | exp=0
}

function randn(m, s) {
    var a = 1 - Math.random();
    var b = 1 - Math.random();
    var c = Math.sqrt(-2 * Math.log(a));
    if (0.5 - Math.random() > 0) {
        return c * Math.sin(Math.PI * 2 * b) * s + m;
    } else {
        return c * Math.cos(Math.PI * 2 * b) * s + m;
    }
}
