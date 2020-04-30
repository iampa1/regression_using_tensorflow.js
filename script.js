/**
 * Get the car data reduced to just the variables we are interested and cleaned of missing data.
 */

async function getData() {
	const carDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
	const carData = await carDataReq.json();
	const cleaned = carData.map( d => ({
		mpg: d.Miles_per_Gallon,
		horsepower: d.Horsepower,
	}))
	.filter( d => d.mpg != null && d.horsepower != null);

	return cleaned;

}

async function run() {
	const data = await getData();
	const values = data.map( m => ({
		x:m.horsepower,
		y:m.mpg,
	}));
	
	tfvis.render.scatterplot( 
		{ name: 'Horsepower v MPG'},
		{ values},
		{
			xLabel : 'Horsepower',
			yLabel : 'Miles per Gallon',
			height : 300
		}
	);

	const model = createModel();
	tfvis.show.modelSummary({name:'Model Summary'},model);
}


function createModel() {
	const model = tf.sequential();

	model.add(tf.layers.dense({inputShape:[1], units:1, useBias:true}));


	model.add(tf.layers.dense({units:1,useBias: true}));

	return model;
}



/**
 * Convert the input data to tensors that we can use for machine learning. We will also do the important best practices of _shuffling_ the data and _normalizing_ the data 
 */

function convertToTensor(data) {
	// Wrapping these calculations in a tidy will dispose of any intermediate tensors.
	
	return tf.tidy( () => {
		//Step 1. Shuffle the data 
		tf.util.shuffle(data);
		
		//Step 2. Convert data to Tensor
		const inputs =data.map(d => d.horsepower);
		const values =data.map(d => d.mpg);


		const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
		const labelTensor = tf.tensor2d(labels, [labels.length, 1]);


		//Step 3. Normalize the data to the range 0-1 using min-max scaling
		const inputMax = inputTensor.max();
		const inputMin = inputTensor.min();
		const labelMax = labelTensor.max();
		const labelMin = labelTensor.min();

		const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
		const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

		return {

			inputs: normalizedInputs,
			labels: normalizedLabels,

			//Return the min/max bounds so we can use them later.
			
			inputMax,
			inputMin,
			labelMax,
			labelMin,
		}
	});
}

		

document.addEventListener('DOMContentLoaded',run);
