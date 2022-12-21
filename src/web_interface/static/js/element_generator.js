function entityElement(iri) {
    if(iri.includes('fkg.iust.ac.ir')) {
        splitted = iri.split('/')
        entity_name = splitted[splitted.length - 1].replace(/_/g, " ").replace(/ـ/g, " ");
        innerElement = `<a target="_blank" href='${iri}'>${entity_name}</a>`
    }
    else {
        innerElement = iri.replace(/_/g, " ").replace(/ـ/g, " ");
    }


    return `
            <p>
                ${innerElement}
            </p>`
}

function reformQuery(query){
    return query.replace(/&/g, "&amp;").replace(/>/g, "&gt;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
}

function singleAnswerElement(rel, prob, probStr, entityPartResult, query=undefined, qWord=undefined, qIri=undefined){
    if($("#show_query_cb").is(':checked'))
        initial_outer_query_style = ''
    else
        initial_outer_query_style = '"display: none;"'

    isTransparent = false;
    if(probStr == undefined){
        titleAlign = "center";
    }
    else{
        titleAlign = "right";
        if(prob < 30) isTransparent = true;
    }

    if(qWord != undefined){
        if(qIri == undefined) questionEntityElement = `${qWord}`;
        else questionEntityElement = `<a target="_blank" href=${qIri}>${qWord}</a>`;
    }

    return `
            <div class='answer-container ${isTransparent ? "transparent": ""}'>
                <div class="header">
                    <div class="title" style='text-align: ${titleAlign}'>
                        <p> رابطه‌ی تشخیص داده شده
                            <i>
                                &nbsp;&quot;
                                ${rel}
                                &quot;
                            </i>
                        </p>
                        ${ qWord == undefined ? '' :
                            `<p> برای موجودیت
                                <i>
                                    &nbsp;&quot;
                                    ${questionEntityElement}
                                    &quot;
                                </i>
                            </p>`
                        }
                    </div>
                    ${ probStr == undefined ? '' :
                        `<div class="prob">
                            <p>احتمال ${probStr}%</p>
                        </div>`
                     }
                </div>
                <div class="card">
                    <div class="entity-container">
                        ${entityPartResult}
                    </div>
                </div>
                ${ query == undefined ? '' :
                    `
                    <div class="query-outer" style=${initial_outer_query_style}>
                        <div class="query-title">
                            <p>کوئری ارسال شده </p>
                        </div>
                        <div class="query-inner">
                            <code>${reformQuery(query)}</code>
                        </div>
                    </div>
                    `
                }
            </div>`
}

/**
  * dict: [{'rel', 'prob'?, 'result': *result_list}]
  *     * result_list = [{'q_word':single_ent_word, 'q_iri': single_ent_iri, 'answers': **answers, 'query': query?)}]
  *         ** answers' length is at least 1
  */
function answerElements(dict) {
    rel = dict.rel
    prob = dict.prob
    resultList = dict.result

    probStr = prob;
    if(prob != undefined){
        prob *= 100
        probStr = prob.toString().substring(0, 3).replace(/\d/g, c => '۰۱۲۳۴۵۶۷۸۹'[c])
        l = probStr.length
        if (probStr[l-1] == '.')
            probStr = probStr.substring(0, l-1)
    }

    if(resultList.length == 0){
        entityPartResult = `<p class="error">هیچ موجودیتی یافت نشد :(</p>`;
        return singleAnswerElement(rel, prob, probStr, entityPartResult);
    }


    return resultList.map(function(result) {
        qWord = result.q_word;
        qIri = result.q_iri;
        answers = result.answers;
        query = result.query;

        answerEntities = answers.map(function(iri) {
            return entityElement(iri)
        });
        entityPartResult = answerEntities.join("\n");

        return singleAnswerElement(rel, prob, probStr, entityPartResult, query, qWord, qIri);
    })

}

function showEntitiesInfo(resp) {
    detectedWords = resp.detected_words;
    detectedEntities = resp.detected_entities;

    if(detectedWords != undefined) {
        $("#entities_info").css("display", "block");

        allWords = new Array();
        detectedWords.forEach(function(item, index) {
            allWords= allWords.concat(`<p>${item}</p>`);
        });
        $("#detected_words").html(allWords);

        allEntities = new Array();
        detectedEntities.forEach(function(item, index) {
            allEntities = allEntities.concat(`<p><a target="_blank" href=${item.iri}>${item.word}</a></p>`);
        });
        $("#detected_entities").html(allEntities);
    }
}

function responseAnswerElement(respAns) {
    if (Array.isArray(respAns)){ // multiple_relation state is on
        allAnsElements = new Array();
        respAns.forEach(function(item, index) {
            ansElements = answerElements(item);
            allAnsElements = allAnsElements.concat(ansElements);
        });
        return allAnsElements;
    }
    else {
        return answerElements(respAns)
    }
}

function errorElement(error) {
    return `
            <div class="request-error">
                <p>${error}</p>
            </div>`
}