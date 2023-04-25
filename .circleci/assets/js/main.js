(function() {
    'use strict';

    var pdfApp = angular.module('pdfApp', ['pdf.viewer', 'oc.lazyLoad']);

    pdfApp.config(['pdfViewerServiceProvider',
        function(pdfViewerServiceProvider) {
            pdfViewerServiceProvider.setPath('assets/js/pdfjs');
        }
    ]);
})();
(function() {
    'use strict';

    angular.module('pdfApp')
        .component('pdfMain', {
            templateUrl: 'assets/tpl/pdfMain.tpl.html',
            controller: pdfMainCtrl,
            bindings: {}
        });

    pdfMainCtrl.$inject = ['$interval'];
    function pdfMainCtrl($interval) {
        var $ctrl = this;
        var files = [
            'paper-latest.pdf'
        ]

        $ctrl.searchQuery = '';
        $ctrl.fileUrl = files[0];
        $ctrl.toggleNext = false;
        $ctrl.togglePrevious = false;

        $ctrl.nextMatch = function() {
            $ctrl.toggleNext = !$ctrl.toggleNext;
        };
        $ctrl.previousMatch = function() {
            $ctrl.togglePrevious = !$ctrl.togglePrevious;
        };
        $ctrl.search = function(query) {
            $ctrl.searchQuery = query;
            console.log('Search Query from Main: ', $ctrl.searchQuery);
        };
        $ctrl.viewerUpdated = function() {
            console.log('Viewer has been updated');
        };
    }
})();
(function() {
    'use strict';

    angular.module('pdfApp')
        .component('pdfTools', {
            templateUrl: 'assets/tpl/pdfTools.tpl.html',
            controller: pdfToolsCtrl,
            bindings: {
                search: '&',
                fullscreen: '&',
                highlight: '&',
                next: '&',
                previous: '&',
            }
        });

    function pdfToolsCtrl() {
        var $ctrl = this;

        $ctrl.searchQuery = '';
        $ctrl.highlightAll = false;
        $ctrl.keywords = ['nonlinear', 'discuss', 'duffing', 'equation'];

        $ctrl.selectKeyword = function(index) {
            $ctrl.searchQuery = $ctrl.keywords[index];
            $ctrl.search({
                query: $ctrl.searchQuery
            });
        };
        $ctrl.onChecked = function() {
            $ctrl.highlight({
                highlightAll: $ctrl.highlightAll
            });
        };
    }
})();
