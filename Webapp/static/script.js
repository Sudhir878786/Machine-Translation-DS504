"use strict";

$('document').ready(() => {

    const ENDPOINT = window.location.href;

    var sourceLangOption = $("#source-lang-dropdown")
    var targetLangOption = $("#target-lang-dropdown")
    var sourceTextbox = document.getElementById('source-text-txtbox')
    var targetTextbox = document.getElementById('target-text-txtbox');

    var selectedSourceLang = sourceLangOption.data("value");
    var selectedTargetLang = targetLangOption.data("value");

    var translator = new OnDemandTranslator(ENDPOINT,
        translatedText => {

            if (translator.hasDataToSend()) {
                targetTextbox.value = "";
                targetTextbox.placeholder = "Translating...";

            } else {
                targetTextbox.placeholder = "";
                targetTextbox.value = translatedText;
            }

        }, error => {
            console.error(error);
        }
    );
    var languageDetector = new OnDemandLanguageDetector(ENDPOINT, language => {
        console.log(`Predicted language: ${language}`);
        
        let sourceLang = language;
        let targetLang = language == "en" ? "fr" : "en";

        // Set the text of "auto" to "English (Detected)" or "French (Detected)"
        if (sourceLang == "en") {
            changeSelectedOptionInDropdown(sourceLangOption, "auto", "English (Detected)");
            changeSelectedOptionInDropdown(targetLangOption, "fr", "French");

        } else {
            changeSelectedOptionInDropdown(sourceLangOption, "auto", "French (Detected)");
            changeSelectedOptionInDropdown(targetLangOption, "en", "English");
        }

        // Get the source text
        let sourceText = sourceTextbox.value;

        // Remove the newline character
        sourceText = sourceText.replace(/(\r\n|\n|\r)/gm, "");

        // Replace all double white spaces with single spaces
        sourceText = sourceText.replace(/\s+/g, " ");

        if (sourceText.length == 0) {
            targetTextbox.value = "";
            targetTextbox.placeholder = "";

        } else {
            translator.translate(sourceText, sourceLang, targetLang);
            targetTextbox.placeholder = "Translating...";
            targetTextbox.value = "";
        }

    }, error => {
        console.error(error);
    });

    var loadingModalTime = new Date();
    $('#loadingModal').modal({
        keyboard: false,
        backdrop: "static",
        show: true
    });

    $('#errorModal').modal({
        keyboard: false,
        backdrop: "static",
        show: false
    });

    var getStatusWebService = new GetStatusWebService(ENDPOINT);
        getStatusWebService.waitForStatus(5)
            .then(() => {
                console.log("Success in contacting with server");

                // Note: there is a min. 500 ms after calling modal.show() before calling modal.hide()
                var timeEllapsed = new Date() - loadingModalTime;

                if (timeEllapsed < 500) {
                    setTimeout(() => {
                        $("#loadingModal").modal("hide");
                    }, 500);   

                } else {
                    $("#loadingModal").modal("hide");
                }
            })
            .catch(error => {
                console.error("Error in getting status from server:");
                console.error(error);

                // Note: there is a min. 500 ms after calling modal.show() before calling modal.hide()
                var timeEllapsed = new Date() - loadingModalTime;

                if (timeEllapsed < 500) {
                    setTimeout(() => {
                        $("#loadingModal").modal("hide");
                        $("#errorModal").modal("show");
                    }, 500);   
                                 
                } else {
                    $("#loadingModal").modal("hide");
                    $("#errorModal").modal("show");
                }
            });

    $("#error-modal-reload-pg-button").click(() => {
        location.reload();
    });
    

    $(".dropdown-item").click(function () {

        // Add the value of the dropdown item to the parent's data-value attribute
        var selectedOption = ($(this).data("value"));
        $(this).parents(".dropdown").data("value", selectedOption);

        // Change the text of dropdown to the selected option
        $(this).parents(".dropdown").find('.btn').html($(this).text() + '<span class="caret"></span>');
        $(this).parents(".dropdown").find('.btn').val($(this).data('value'));

        // Set the "active" classname to the selected dropdown item
        $(this).parents(".dropdown-menu").find('a').removeClass('active');
        $(this).addClass('active');

        selectedSourceLang = sourceLangOption.data("value");
        selectedTargetLang = targetLangOption.data("value");
    });

    function changeSelectedOptionInDropdown(dropdownElement, dataValue, dropdownText) {
        // Set the text of the selected option
        dropdownElement.find('.btn').html(dropdownText + '<span class="caret"></span>');
        dropdownElement.find('.btn').val(dataValue);

        console.log(dropdownText);

        // Set the "active" classname to the selected dropdown item
        dropdownElement.find(".dropdown-menu").find('a').removeClass('active');
        dropdownElement.find(".dropdown-item").find(`[data-value='${dataValue}']`).addClass('active');
    }

    sourceTextbox.addEventListener('keyup', function (e) {
        let sourceText = e.target.value;

        // Remove the newline character
        sourceText = sourceText.replace(/(\r\n|\n|\r)/gm, "");

        // Replace all double white spaces with single spaces
        sourceText = sourceText.replace(/\s+/g, " ");

        console.log(selectedSourceLang)

        if (sourceText.length == 0) {
            targetTextbox.value = "";
            
            if (selectedSourceLang == "auto") {
                changeSelectedOptionInDropdown(sourceLangOption, "auto", "Any language");
                changeSelectedOptionInDropdown(targetLangOption, "fr", "French");
            }

        } else if (selectedSourceLang == "auto") {
            languageDetector.predictLanguage(sourceText);
            targetTextbox.value = "";
            targetTextbox.placeholder = "Translating...";

        } else {
            translator.translate(sourceText, selectedSourceLang, selectedTargetLang);
            targetTextbox.value = "";
            targetTextbox.placeholder = "Translating...";
        }
    });
});