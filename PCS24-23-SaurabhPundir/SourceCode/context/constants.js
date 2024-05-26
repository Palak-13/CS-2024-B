//organizerAddress = '0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266'

import voting from "./Create.json";

export const VotingAddress = process.env.NEXT_PUBLIC_DEPLOYED_ADDRESS;
export const VotingAddressABI = voting.abi;
export const organiserAddress = (process.env.NEXT_PUBLIC_ORGANISER).toLowerCase();