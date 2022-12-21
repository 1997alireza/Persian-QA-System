var loading = false;

$(document).ready(function(){
    if(Cookies.get('authentication_id') == "admin"){
        $("#login_div").css("display", "none");
        $("#logout_div").css("display", "block");
        $("#show_query_cb_div").css("display", "block");
    }

    $("#show_query_cb").change(function(){
        if ($(this).is(':checked')) {
            $(".query-outer").show();
        }
        else {
            $(".query-outer").hide();
        }
    });

    $("#question").keypress(function(e){
        if(e.which === 13) requestAnswer();
    });

    $("#rest_answers_cb").change(function(){
        requestAnswer();
    });
});

function requestAnswer(){
    if(!loading) {
        var text = $("#question").val()
        if(text.replace(/\s+/g, '').length == 0) return;
        on_loading();
        var multiple_relations = $("#rest_answers_cb").prop("checked")
        $("#answer").empty();
        $("#entities_info").css("display", "none");
        data = {
            question: text,
            multiple_relations: multiple_relations
        }
        if(Cookies.get('authentication_id')) {
            data.authentication_id = Cookies.get('authentication_id') + "";
            data.authentication_code = Cookies.get('authentication_code') + "";
        }
        $.ajax({
            url: "/answer",
            type: "post",
            cache: false,
            data: data,
            success: function (response) {
                off_loading();
                try {
                    jsonResp = JSON.parse(response);
                    showEntitiesInfo(jsonResp);
                    $("#answer").html(responseAnswerElement(jsonResp.ret_ans));
                }
                catch(err){
                    $("#answer").html(errorElement("درخواست ناموفق، دوباره تلاش کنید..."));
                    console.error(err);
                }
            },
            error: function (jqXHR) {
                off_loading();
                $("#answer").html(errorElement("درخواست ناموفق، دوباره تلاش کنید..."));
                console.error(jqXHR);
            }
        });
    }
}

function login(){
    Cookies.set('authentication_id', 'admin');
    Cookies.set('authentication_code', '0000'); // TODO: must be received from server
    location.reload();
}

function logout(){
    Cookies.remove('authentication_id');
    Cookies.remove('authentication_code');
    location.reload();
}

function on_loading(){
    loading = true;
    $("body").css("cursor", "progress");
    $(".switch-div:has(#rest_answers_cb)").css("cursor", "progress");
    $("#rest_answers_cb").attr("disabled", true);
}

function off_loading(){
    loading = false;
    $("body").css("cursor", "auto");
    $(".switch-div:has(#rest_answers_cb)").css("cursor", "pointer");
    $("#rest_answers_cb").removeAttr("disabled");
}